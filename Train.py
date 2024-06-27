import os
import time
import numpy as np
from math import ceil
from osgeo import gdal, osr, ogr
from scipy.ndimage import distance_transform_edt
import io
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
# from sklearn.metrics import confusion_matrix

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus: #Currently, memory growth needs to be the same across GPUs
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e) #Memory growth must be set before GPUs have been initialized

import matplotlib.pyplot as plt
import random
from lib.UNET import *
from lib.ResUnet import *
from lib.ResUnet_blocks import ResBlock
from lib.HRNET import *
from lib.ResUnet_plus import ResUnetPlusPlus, ResUnetPlusPlus_d3
from lib.unet3plus import *
from lib.SegNet import segnet
from lib.DeepLabV3P import deeplabv3_plus
from lib.U_HRNET import UHRNet, UHRNet_W18_Small, UHRNet_W18, UHRNet_W48
from lib.DenseNet import densenet_model
from lib.MACUNET import MACUNet
import psutil  # 用于获取内存使用情况


min_label_cpunt = 0
# 预处理nan值
def fill_nans(data_matrix):
    """Fills NaN's with nearest neighbours."""
    if np.any(np.isnan(data_matrix)):
        indices = distance_transform_edt(
            np.isnan(data_matrix), return_distances=False, return_indices=True
        )
        data_matrix = np.where(np.isnan(data_matrix), data_matrix[tuple(indices)], data_matrix)
    return data_matrix

# some metrics or Loss
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def focal_loss(gamma=2.0, alpha=0.25):
    """
    创建一个focal loss函数。
    :param gamma: 调节易分类样本权重的指数。
    :param alpha: 平衡正负样本的权重。
    :return: focal loss函数
    """
    def focal_loss_fixed(y_true, y_pred):
        """
        Focal loss的实际计算。
        :param y_true: 真实标签。
        :param y_pred: 预测标签。
        :return: 计算的loss值。
        """
        # 将预测值限制在epsilon和1-epsilon之间，以防止计算log时出现NaN或Inf
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # 计算focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    return focal_loss_fixed

def mcc_m(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn) - (fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    mcc = numerator / (denominator + K.epsilon())
    return mcc

def generate_validation_data(generator, steps):
    while True:
        for _ in range(steps):
            yield next(generator)

class TrainingLogger(Callback):
    def __init__(self, log_file, batch_size, filters_root, learning_rate, model_name, 
                 img_train_csv, img_val_csv, img_path_train, img_path_val, bands, bandlist):
        super(TrainingLogger, self).__init__()
        self.log_file = log_file
        self.batch_size = batch_size
        self.filters_root = filters_root
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.img_train_csv = img_train_csv
        self.img_val_csv = img_val_csv
        self.img_path_train = img_path_train
        self.img_path_val = img_path_val
        self.bands = bands
        self.bandlist = bandlist
        self.start_time = time.time()
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        total_time = time.time() - self.start_time
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # Memory usage in MB

        with open(self.log_file, 'a') as f:
            f.write(f"Epoch: {epoch+1}, Duration: {epoch_duration:.2f}s, Total Time: {total_time:.2f}s, Memory Usage: {memory_usage:.2f}MB\n")

    def on_train_begin(self, logs=None):
        with open(self.log_file, 'w') as f:
            f.write(f"Training Model: {self.model_name}\n")
            f.write(f"Batch Size: {self.batch_size}, Filters Root: {self.filters_root}, Learning Rate: {self.learning_rate}\n")
            f.write(f"Train CSV: {self.img_train_csv}, Validation CSV: {self.img_val_csv}\n")
            f.write(f"Bands: {self.bands}\n")
            f.write(f"Band List: {self.bandlist}\n")
            f.write(f"Train Image Path: {self.img_path_train}, Validation Image Path: {self.img_path_val}\n")
            f.write("Epoch, Duration(s), Total Time(s), Memory Usage(MB)\n")

            # 模型结构的记录
            stream = io.StringIO()
            self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
            model_summary = stream.getvalue()
            stream.close()
            f.write("\nModel Summary:\n" + model_summary + "\n")

class CustomModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_f1_m', save_best_only=True, mode='max', start_epoch=30, min_epoch_interval=5):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.start_epoch = start_epoch
        self.min_epoch_interval = min_epoch_interval
        self.best = None
        self.last_saved_epoch = -min_epoch_interval  # Ensure the first save is allowed after start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_epoch:
            return
        
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if (self.best is None or 
            (self.mode == 'max' and current > self.best) or 
            (self.mode == 'min' and current < self.best)) and (epoch - self.last_saved_epoch >= self.min_epoch_interval):
            self.best = current
            self.model.save(self.filepath.format(epoch=epoch+1, **logs))
            self.last_saved_epoch = epoch
            print(f'\nEpoch {epoch + 1}: saving model to {self.filepath}')

class UpdateLRLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            logs = {}
        
        lr = self.model.optimizer.learning_rate
        
        # Check if lr is a learning rate schedule
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = lr(tf.cast(self.model.optimizer.iterations, tf.float32))
        else:
            current_lr = tf.keras.backend.get_value(lr)
        
        logs['lr'] = float(current_lr)  # Update logs dictionary with the learning rate
        print(f"\nEpoch {epoch + 1}: Learning rate is {current_lr}")

class TimeHistory(Callback):
    def __init__(self):
        super().__init__()
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_start_time)
        tf.print("Epoch {} finished in {:.2f} seconds".format(epoch, self.times[-1]))

class CropTrain():

    def __init__(self) -> None:
        self.trainDataDict = {}

    # save raster to disk
    def SaveRaster(self, fileName, proj, geoTrans, data):

        # type
        if 'int8' in data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # check shape of array
        if len(data.shape) == 3:
            im_height, im_width, im_bands = data.shape
        else:
            im_bands, (im_height, im_width) = 1, data.shape 

        # create file
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(fileName, im_width, im_height, im_bands, datatype, options=['COMPRESS=LZW'])
        if len(geoTrans) == 6:
            dataset.SetGeoTransform(geoTrans)
        if len(proj) > 0:
            dataset.SetProjection(proj)

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(data)
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(data[:, :, i])

    # get information of the raster
    def LoadRasterInfo(self, rasterFile, bandNum=1):
        # Open the file:
        dataset = gdal.Open(rasterFile)
        band = dataset.GetRasterBand(bandNum).ReadAsArray()
        geoTrans = dataset.GetGeoTransform()
        proj = dataset.GetProjection()
        rows, cols = band.shape

        return proj, geoTrans, rows, cols
    
    # read raster data
    def LoadRaster(self, rasterFile, bandN=7, bands=[1, 2, 3, 4, 5, 6, 7]):           
        # Open the file:
        dataset = gdal.Open(rasterFile)

        # Label
        if dataset.RasterCount == 1:
            band = dataset.GetRasterBand(1).ReadAsArray()
            return band
        
        # Get dimensions from the first band
        first_band = dataset.GetRasterBand(bands[0]).ReadAsArray()
        rows, cols = first_band.shape
        
        # Initialize the first band and reshape
        band = first_band.reshape((rows, cols, 1))

        # Read and concatenate the rest of the bands
        for bandNum in bands[1:]:
            tmpBand = dataset.GetRasterBand(bandNum).ReadAsArray().reshape((rows, cols, 1))
            band = np.concatenate([band, tmpBand], axis=2)

        band = fill_nans(band)  # fill nodata
        return band


    def get_train_val_random(self, imgpath, val_rate = 0.3):
        train_url = []    
        train_set = []
        val_set  = []

        for pic in os.listdir(imgpath):
            if pic.endswith('.tif'):
                train_url.append(pic)

        random.seed(57)
        random.shuffle(train_url)

        total_num = len(train_url)
        val_num = int(val_rate * total_num)
        for i in range(len(train_url)):
            if i < val_num:
                val_set.append(train_url[i]) 
            else:
                train_set.append(train_url[i])
        return train_set,val_set
    
    def get_Train_val_specify(self, imgpathTrain, imgpathVal):
        if not os.path.exists(imgpathTrain):
            print(f"Label file not found: {imgpathTrain}")
            return
        if not os.path.exists(imgpathVal):
            print(f"Label file not found: {imgpathVal}")
            return
        
        train_set = []
        val_set  = []

        for pictrain in os.listdir(imgpathTrain):
            if pictrain.endswith(".tif"):
                train_set.append(pictrain)

        for picval in os.listdir(imgpathVal):
            if picval.endswith(".tif"):
                val_set.append(picval)            
        return train_set, val_set


    def get_train_val_from_csv(self, train_csv_path, val_csv_path):
        if not os.path.exists(train_csv_path):
            print(f"Training CSV file not found: {train_csv_path}")
            return [], []
        if not os.path.exists(val_csv_path):
            print(f"Validation CSV file not found: {val_csv_path}")
            return [], []

        train_images = []
        # train_labels = []
        val_images = []
        # val_labels = []

        # 读取训练集CSV文件
        with open(train_csv_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                train_images.append(row[0])  # 第一列是影像路径
                # train_labels.append(row[1])  # 第二列是标签路径

        # 读取验证集CSV文件
        with open(val_csv_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                val_images.append(row[0])  # 第一列是影像路径
                # val_labels.append(row[1])  # 第二列是标签路径

        return train_images, val_images   


    def constructLabelFilename(self, imgFilename):
        # 分割文件名以删除日期部分
        parts = imgFilename.split('_')
        if len(parts) > 3:  # 确保文件名有足够的部分来分割
            # 移除日期部分并重新构造文件名
            labelFilename = '_'.join(parts[:-1]) + '.tif'
        else:
            raise ValueError("图像文件名格式不正确: " + imgFilename)
        return labelFilename    
    
    # data for training
    def generateData(self, imgpath, maskpath, batch_size, data=[], bandN=7, bands=[1, 2, 3, 4, 5, 6, 7]):

        if bandN != len(bands):
            raise ValueError(f"bandN ({bandN}) and the length of bands ({len(bands)}) must be the same.")

        while True:
            imgTrain = []
            labelTrain = []
            # !encodingTrain = []  # 新增一个列表来存储编码
            flagB = True
            batch = 0

            for i in range(len(data)):
                url = data[i]
                imgData = self.LoadRaster(imgpath + url, bands=bands)

                labelUrl = self.constructLabelFilename(url)  # 构造对应的标签图像文件名
                labelData = self.LoadRaster(maskpath + labelUrl) # 加载标签数据

                batch += 1

                imgData = np.expand_dims(imgData, axis=0)
                imgData = imgData.astype('float32')
                labelData = np.expand_dims(labelData, axis=0)
                labelData = np.expand_dims(labelData, axis=3)
                labelData = labelData.astype('float32')

                # 处理编码数据
                # !encoding = np.expand_dims(encoding, axis=0)

                if flagB:
                    imgTrain = imgData
                    labelTrain = labelData
                    # !encodingTrain = encoding  # 保存编码
                    flagB = False
                else:
                    imgTrain = np.concatenate([imgTrain, imgData])
                    labelTrain = np.concatenate([labelTrain, labelData])
                    #! encodingTrain = np.concatenate([encodingTrain, encoding])

                if batch % batch_size == 0:
                    # ! yield ([imgTrain, encodingTrain], labelTrain)  # 将编码作为额外的输入传递
                    yield (imgTrain, labelTrain)
                    imgTrain = []
                    labelTrain = []
                    # ! encodingTrain = []
                    flagB = True

    # data for validation
    def generateValidData(self, imgpath, maskpath, batch_size,data=[], bandN=7, bands=[1, 2, 3, 4, 5, 6, 7]):  
        if bandN != len(bands):
            raise ValueError(f"bandN ({bandN}) and the length of bands ({len(bands)}) must be the same.")

        while True:  
            imgTrain = []  
            labelTrain = []
            # ! encodingTrain = []  # 新增一个列表来存储编码
            flagB = True
            batch = 0  

            for i in (range(len(data))): 
                url = data[i]
                imgData = self.LoadRaster(imgpath + url, bands=bands)
                labelUrl = self.constructLabelFilename(url)  # 构造对应的标签图像文件名
                labelData = self.LoadRaster(maskpath + labelUrl) # 加载标签数据

                # if np.sum(labelData == 1) < min_label_cpunt:
                #     continue

                batch += 1
                imgData = np.expand_dims(imgData, axis = 0)
                imgData = imgData.astype('float32')
                labelData = np.expand_dims(labelData, axis = 0)
                labelData = np.expand_dims(labelData, axis = 3)
                labelData = labelData.astype('float32')

                # 处理编码数据
                # ! encoding = np.expand_dims(encoding, axis=0)

                if flagB:
                    imgTrain = imgData
                    labelTrain = labelData
                    # ! encodingTrain = encoding  # 保存编码
                    flagB = False
                else:
                    imgTrain = np.concatenate([imgTrain, imgData])
                    labelTrain = np.concatenate([labelTrain, labelData])
                    # ! encodingTrain = np.concatenate([encodingTrain, encoding])

                if batch % batch_size == 0:
                    # ! yield ([imgTrain, encodingTrain], labelTrain)  # 将编码作为额外的输入传递
                    yield (imgTrain, labelTrain)
                    imgTrain = []
                    labelTrain = []
                    # ! encodingTrain = []
                    flagB = True

    def createModel(self, modelName, batch_size, pixelSize, bands, n_label, filters_root=32):
        if modelName == 'unet_d4':
            model = unet_d4(batch_size, pixelSize, pixelSize, bands, n_label)

        elif modelName == 'attunet_d4':
            model = attunet_d4(batch_size, pixelSize, pixelSize, bands, n_label)

        elif modelName == 'resunet':
            model = ResUNet(pixelSize, pixelSize, bands, n_label, filters_root, depth=3)
        
        elif modelName == 'resunet_plus_d4':
            filters = [filters_root, filters_root*2, filters_root*4, filters_root*8, filters_root*16]
            tf.print(f"filters:{filters}")
            model = ResUnetPlusPlus(bands, filters)

        elif modelName == 'UNet_3Plus':
            model = unet3plus(pixelSize, pixelSize, bands, n_label, filters_root)

        elif modelName == 'MACUNet':
            print(f"model is MACUNET")
            model = MACUNet(band_num=bands, class_num=n_label)

        elif modelName == 'segnet':
            model = segnet(pixelSize, pixelSize, bands, classes=n_label)

        elif modelName == 'DeepLabV3Plus':
            model = deeplabv3_plus(shape=(pixelSize, pixelSize, bands))

        elif modelName == 'UHRNet_W48':
            model = UHRNet_W48(in_channels=bands, nclasses=n_label)

        elif modelName == 'DenseNet':
            model = densenet_model(classes=n_label, shape=(pixelSize, pixelSize, 3), batch_size=batch_size)
        
        elif modelName == 'MACUNet':
            model = MACUNet(band_num=bands, class_num=n_label)

        elif modelName.startswith('HRNet'):
            input_shape = (pixelSize, pixelSize, bands)
            tf.print(f"input_shape:{input_shape}")

            channels_map = {
                18: [18, 36, 72, 144],  # 对应HRNet_W18的配置
                24: [24, 48, 96, 192],  # 对应HRNet_W24的配置
                32: [32, 64, 128, 256], # 对应HRNet_W32的配置
                40: [40, 80, 160, 320], # 对应HRNet_W40的配置
                48: [48, 96, 192, 384], # 对应HRNet_W48的配置
            }

            channels = channels_map.get(filters_root)
            # channels = 32
            tf.print(f"channels:{channels}")
            if not channels:
                raise ValueError(f"Unsupported filters_root for HRNet: {filters_root}")
            model = HRNet(channel_list=channels,
                        input_shape=input_shape,
                        num_classes=n_label+1,
                        weights=None,
                        name=f'HRNet_W{filters_root}')

        else:
            raise ValueError("未知的模型名称: {}".format(modelName))

        return model

    def trainCropModel(self, imgTrainCSV, imgValCSV, imgPathTrain, imgPathVal, maskPathTrain, maskPathVal, modelSavePath, modelName, pixelSize=256, n_label=1, 
                       epochs=50, batch_size=16, bands=7, bandlist=[1, 2, 3, 4, 5, 6, 7], filters_root=32, existModel=False, existing_model_path=None, additional_epochs=0):
        
        if bands != len(bandlist):
            print(f"bandN ({bands}) and the length of bands ({len(bandlist)}) must be the same.")
            return None

        # Prepare data
        train_set, val_set = self.get_train_val_from_csv(imgTrainCSV, imgValCSV)
        train_numb = len(train_set)
        valid_numb = len(val_set)
        print("train_numb:", train_numb, ",  valid_numb:", valid_numb)

        # Prepare model saving directory and checkpoint
        model_name = modelName + '_%s_m.{epoch:03d}.tf' % 't'
        filepath = os.path.join(modelSavePath, model_name)
        # checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_f1_m', verbose=1, save_best_only=True, mode='max')
        custom_checkpoint = CustomModelCheckpoint(filepath=filepath, monitor='val_f1_m', save_best_only=True, mode='max', start_epoch=30, min_epoch_interval=5)
        # 多GPU训练
        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            # Choose to train a new model or continue training an existing model
            if not existModel:
                model = self.createModel(modelName, batch_size, pixelSize, bands, n_label, filters_root)
            else:
                # Load existing model
                custom_objects = {
                                'ResBlock': ResBlock, 
                                'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m, 'dice_coef_loss': dice_coef_loss, 'mcc_m': mcc_m}
                model = load_model(existing_model_path,custom_objects)
                epochs += additional_epochs  # Adjust total number of epochs

            # 线性调整学习率
            lr = 1e-3
           
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.001, cooldown=10, min_lr=1e-7)
            optimizer = Adam(learning_rate=lr)

        # Prepare loggers and callbacks
        update_lr_logger = UpdateLRLogger()
        csv_logger = CSVLogger(os.path.join(modelSavePath, csvname), append=True)
        log_file_path = os.path.join(modelSavePath, txtname)
        training_logger = TrainingLogger(log_file=log_file_path, 
                                        batch_size=batch_size, 
                                        filters_root=filters_root, 
                                        learning_rate=initial_learning_rate, 
                                        model_name=modelName,
                                        img_train_csv=imgTrainCSV, 
                                        img_val_csv=imgValCSV, 
                                        img_path_train=imgPathTrain, 
                                        img_path_val=imgPathVal,
                                        bands=bands,
                                        bandlist=bandlist)   

        # Training process
        bgTime = time.time()
        history = model.fit(
            self.generateData(imgPathTrain, maskPathTrain, batch_size, train_set, bandN=bands, bands=bandlist),
            steps_per_epoch=train_numb // batch_size,
            epochs=epochs,
            validation_data=self.generateValidData(imgPathVal, maskPathVal, batch_size, val_set, bandN=bands, bands=bandlist),
            validation_steps=valid_numb // batch_size,
            callbacks=[custom_checkpoint, update_lr_logger, csv_logger, training_logger],
            initial_epoch=0 if not existModel else epochs - additional_epochs
        )

        tf.print('Training completed. Time elapsed:', (time.time() - bgTime) / 60, 'min')

if __name__ == '__main__':

    cropTrain = CropTrain()

    # dimension of train samples
    pixelsize = 256
    # number of class type
    nlabel = 1

    #02_ModelSave_23Train6S_23Test2S
    imgTrainCSV  = r"/root/Train.csv"
    imgValCSV    = r"/root/Test.csv"
    imgPathTrain = r'/root/Train/'
    imgPathVal   = r'/root/Test/'

    maskPathTrain = r'/root/Label/'
    maskPathVal   = r'/root/Label/'

    # train the model
    modelname = 'UNet_3Plus'
    modelSavePath =f'/root/{modelname}'
    txtname = f'{modelname}.txt'
    csvname = f'{modelname}.csv'
    
    cropTrain.trainCropModel(imgTrainCSV, imgValCSV, imgPathTrain, imgPathVal, maskPathTrain, maskPathVal, modelSavePath, modelname, 
                             pixelSize=pixelsize, n_label=nlabel, epochs=100, batch_size=8, bands=5, bandlist=[1, 2, 3, 4, 6], filters_root=64,
                            #  existModel=True, existing_model_path=modelpath, additional_epochs=85
                             )
