from tensorflow.keras.layers import *
from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D,  Dropout, concatenate, Add, UpSampling2D,Dense
from keras.optimizers import Adam
from keras.layers import Activation,Reshape,Add,Concatenate,Lambda,multiply
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D,MaxPooling2D
import keras.backend as K
from .PAAB import PAAB
import tensorflow as tf
# def activation(x, func='relu'):
#     '''
#     Activation layer.
#     '''
#     return Activation(func)(x)


class PixelShuffle(Layer):
    def __init__(self, upscale_factor, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.upscale_factor = upscale_factor

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.upscale_factor)

    def get_config(self):
        config = super(PixelShuffle, self).get_config()
        config.update({"upscale_factor": self.upscale_factor})
        return config

class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
        self.scale = self.add_weight(name='scale',
                                     shape=(),
                                     initializer='ones',
                                     trainable=True)

    def call(self, inputs):
        return inputs * self.scale
    
class ResBlock(Layer):
    """
    Represents the Residual Block in the ResUNet architecture.
    """
    def __init__(self, filters, strides, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.filters = filters
        self.strides = strides

        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.conv1 = Conv2D(filters=filters, kernel_size=3, strides=strides, padding="same", use_bias=False)

        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()
        self.conv2 = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False)

        self.conv_skip = Conv2D(filters=filters, kernel_size=1, strides=strides, padding="same", use_bias=False)
        self.bn_skip = BatchNormalization()

        self.add = Add()

    def call(self, inputs, training=False, **kwargs):
        x = inputs
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.conv2(x)

        skip = self.conv_skip(inputs)
        skip = self.bn_skip(skip, training=training)

        res = self.add([x, skip])
        return res

    def get_config(self):
        return dict(filters=self.filters, strides=self.strides, **super(ResBlock, self).get_config())

class ResBlock_PAAB(Layer):
    def __init__(self, filters, strides, **kwargs):
        super(ResBlock_PAAB, self).__init__(**kwargs)
        self.filters = filters
        self.strides = strides

        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.conv1 = Conv2D(filters=filters, kernel_size=3, strides=strides, padding="same", use_bias=False)

        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()
        self.conv2 = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False)

        self.paab = PAAB()        
        # 3*3 或 1*1 都有待验证
        self.conv3 = Conv2D(filters=filters, kernel_size=3, padding='same', use_bias=False)
        self.bn3 = BatchNormalization()
        self.relu3 = ReLU()

        self.conv_skip = Conv2D(filters=filters, kernel_size=1, strides=strides, padding="same", use_bias=False)
        self.bn_skip = BatchNormalization()

        self.add = Add()

    def call(self, inputs, training=False, **kwargs):
        x, S = inputs  # 现在接受一个包含两个元素的列表：特征图x和样式特征S
        # tf.print("inputs shape:", inputs.shape)
        tf.print("S input[1] shape:", S.shape)
        tf.print("x intput[0] shape:", x.shape)

        x1 = self.bn1(x, training=training)
        x1 = self.relu1(x1)
        x1 = self.conv1(x)

        x1 = self.bn2(x1, training=training)
        x1 = self.relu2(x1)
        x2 = self.conv2(x1)

        # PAAB调整
        x_paab = self.paab([S, x2])
        x_combined = Concatenate()([x2, x_paab])  # Concatenate调整后的和原始的

        # 调整通道数
        x_combined = self.conv3(x_combined)
        x_combined = self.bn3(x_combined)
        x_combined = self.relu3(x_combined)

        skip = self.conv_skip(x)
        skip = self.bn_skip(skip, training=training)

        res = self.add([x_combined, skip])
        return res

    def get_config(self):
        return dict(filters=self.filters, strides=self.strides, **super(ResBlock_PAAB, self).get_config())