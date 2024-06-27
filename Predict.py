import os
import time
import numpy as np
from math import ceil
from osgeo import gdal, osr, ogr
from scipy.ndimage import distance_transform_edt

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

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
from lib.ResUnet_blocks import ResBlock
from lib.HRNET import *
from lib.ResUnet_plus import ResUnetPlusPlus
from lib.unet3plus import *
from lib.SegNet import segnet
from lib.DeepLabV3P import deeplabv3_plus
from lib.U_HRNET import UHRNet, UHRNet_W18_Small, UHRNet_W18, UHRNet_W48
from lib.DenseNet import densenet_model
from lib.MACUNET import MACUNet
from lib.HRNET import *

# true samples pixels in one patch
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

class CropPredict():
    def __init__(self) -> None:
        self.predictDataDict = {}

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

    def LoadRasterWindow(self, dataset, x, y, width, height, bands_to_process, throldnan):
        output_data = []
        # 裁剪和处理图像
        for index in bands_to_process:
            band = dataset.GetRasterBand(index).ReadAsArray(x, y, width, height)
            if (np.sum(band == 0) > throldnan) or (np.sum(np.isnan(band))> throldnan):
                return None
            band_date_scale = band * 0.0000275 - 0.2
            if index <= 4:
                band_data_clip = np.clip(band_date_scale, 0, 0.3)
                normalized_band = band_data_clip / 0.3
            else:
                band_data_clip = np.clip(band_date_scale, 0, 0.5)
                normalized_band = band_data_clip / 0.5
            output_data.append(normalized_band.reshape((height, width, 1)))

        band_data = np.concatenate(output_data, axis=2)
        band_data = fill_nans(band_data)  

        return band_data
    # used fo predict
    def PredictCrop(self, modelFile, inFiles, resultPath, bands_to_process, pixelSize=256, overlap=128, drop=32, threshold_nan=5000, batch_size=32):
        if not os.path.exists(os.path.dirname(resultPath)):
            os.makedirs(os.path.dirname(resultPath))
        if not os.path.exists(resultPath):
            os.makedirs(resultPath)
        if not os.path.exists(modelFile):
            print("model file doesn't exist!")
            return None, None
        if isinstance(inFiles, str):
            inFiles = [inFiles]

        # Load model
        custom_objects = {'ResBlock': ResBlock, 'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m, 'dice_coef_loss': dice_coef_loss, 'mcc_m': mcc_m}
        try:
            model = load_model(modelFile, custom_objects=custom_objects) if isinstance(modelFile, str) else modelFile
        except TypeError as e:
            print(f"Error while loading the model: {str(e)}")
            return

        for inFile in inFiles:
            startTime = time.time()

            dataset = gdal.Open(inFile, gdal.GA_ReadOnly)
            x_size, y_size = dataset.RasterXSize, dataset.RasterYSize
            resultFile = os.path.join(resultPath, 'predict_' + os.path.basename(inFile))

            result_accumulator = np.zeros((y_size, x_size), dtype='float32')
            count_matrix = np.zeros((y_size, x_size), dtype='float32')

            batch_tiles = []
            positions = []

            # Batch processing
            for x in range(0, x_size-pixelSize+1, pixelSize - overlap):
                for y in range(0, y_size-pixelSize+1, pixelSize - overlap):
                    tile = self.LoadRasterWindow(dataset, x, y, pixelSize, pixelSize, bands_to_process, threshold_nan)
                    if tile is not None:
                        batch_tiles.append(tile)
                        positions.append((x, y, tile.shape[0], tile.shape[1]))  # Also store original tile size

                        if len(batch_tiles) >= batch_size:
                            self.process_batch(model, batch_tiles, positions, result_accumulator, count_matrix, drop)
                            batch_tiles, positions = [], []

            # Process the remaining batch
            if batch_tiles:
                self.process_batch(model, batch_tiles, positions, result_accumulator, count_matrix, drop)

            final_result = self.compute_final_result(result_accumulator, count_matrix)
            self.SaveRaster(resultFile, dataset.GetProjection(), dataset.GetGeoTransform(), final_result)
            print('Done processing', resultFile, f"in {(time.time() - startTime) / 60} minutes")

    def process_batch(self, model, batch_tiles, positions, result_accumulator, count_matrix, drop):
        # Prepare the batch
        batch_tiles_array = np.stack(batch_tiles, axis=0)
        predictions = model.predict(batch_tiles_array)

        # Iterate over predictions and update the accumulator and count matrices
        for i, (x, y, tile_height, tile_width) in enumerate(positions):
            yPredict = predictions[i].reshape(tile_height, tile_width)
            yPredict_cropped = yPredict[drop:-drop, drop:-drop] if drop > 0 else yPredict

            sy, sx = y + drop, x + drop
            ey, ex = sy + yPredict_cropped.shape[0], sx + yPredict_cropped.shape[1]

            result_accumulator[sy:ey, sx:ex] += yPredict_cropped
            count_matrix[sy:ey, sx:ex] += 1

    def compute_final_result(self, accumulator, count_matrix):
        with np.errstate(divide='ignore', invalid='ignore'):
            final_result = np.divide(accumulator, count_matrix, out=np.zeros_like(accumulator), where=count_matrix != 0)
        return (final_result > 0.5).astype('uint8')

if __name__ == '__main__':

    cropPredict = CropPredict()

    pixelsize = 256
    overlap=128
    drop=32
    nanvalues = 256*256*1.0
    batchsize = 2

    year = 2020
    # 读取CSV文件并获取所有的系统索引
    system_indices = set()
    with open(r'/home/Scenes.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            system_indices.add(row['system_index'])

    inFiles1 = []    
    inpath1 = (r'/home/Scenes')
    for pic in os.listdir(inpath1):
        if pic.endswith('.tif'):
            # 检查基本文件名是否在CSV中的system_indices列
            basename = os.path.splitext(pic)[0]
            if basename in system_indices:
                inFiles1.append(os.path.join(inpath1, pic))

    bands_to_process = [2, 3, 4, 5, 6, 7]  # L8
    resultPath = r'/home/resunet'
    modelFile = r'/home/resunet/resunet_t_m.096.tf'
    log_file = r''
    cropPredict.PredictCrop(modelFile, inFiles1, resultPath, bands_to_process, pixelsize, overlap, drop, nanvalues, batchsize)
