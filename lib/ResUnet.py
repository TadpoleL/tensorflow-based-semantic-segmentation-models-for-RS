from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, UpSampling2D, Concatenate, Activation,Dense
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D,MaxPooling2D
from keras.layers import Activation,Reshape,Add,Concatenate,Lambda,multiply
from tensorflow.keras.models import Model
from .ResUnet_blocks import ResBlock
import math
import keras.backend as K

def activation(x, func='relu'):
    '''
    Activation layer.
    '''
    return Activation(func)(x)


def CBAM(x, ratio=4, attention='both'): # 8
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    if attention == 'channel':
        cbam_feature = CBAM_channel_attention(x, ratio)
    elif attention == 'spatial':
        cbam_feature = CBAM_spatial_attention(x)
    else:
        cbam_feature = CBAM_channel_attention(x, ratio)
        cbam_feature = CBAM_spatial_attention(cbam_feature)

    return cbam_feature

def CBAM_channel_attention(x, ratio=8):
    # channel = x._keras_shape[-1]
    channel = K.int_shape(x)[-1]
    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='glorot_uniform', #he_normal
                             use_bias=True,
                             bias_initializer='zeros')

    shared_layer_two = Dense(channel,
                             kernel_initializer='glorot_uniform',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(x)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(x)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = activation(cbam_feature, 'sigmoid')

    return multiply([x, cbam_feature])


def CBAM_spatial_attention(x):
    kernel_size = 7

    # channel = x._keras_shape[-1]
    channel = K.int_shape(x)[-1]
    cbam_feature = x

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)

    concat = Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    return multiply([x, cbam_feature])


# 在每个编码器块的末尾和解码器块的开始加入了 CBAM 模块
def ResUNet(batch_size, height, width, channel, classes=1, filters_root=64, depth=3):
    """
    Builds ResUNet model with UNet-like input parameters.
    :param batch_size: Batch size of the input images.
    :param height: Height of the input images.
    :param width: Width of the input images.
    :param channel: Number of channels in the input images.
    :param classes: Number of classes that will be predicted for each pixel. Number of classes must be higher than 1.
    :param filters_root: Number of filters in the root block.
    :param depth: Depth of the architecture. Depth must be <= min(log_2(h), log_2(w)).
    :return: Tensorflow model instance.
    """
    input_shape = (height, width, channel)
    if classes < 1:
        raise ValueError("The number of classes must be larger than 1.")
    if not math.log(height, 2).is_integer() or not math.log(width, 2).is_integer():
        raise ValueError(f"Input height ({height}) and width ({width}) must be powers of two.")
    if 2 ** depth > min(height, width):
        raise ValueError(f"Model has insufficient height ({height}) and width ({width}) compared to its desired depth ({depth}).")

    inputs = Input(batch_shape=(batch_size,) + input_shape)

    layer = inputs

    # ENCODER
    encoder_blocks = []

    filters = filters_root
    layer = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")(layer)

    branch = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False)(layer)
    branch = BatchNormalization()(branch)
    branch = ReLU()(branch)
    branch = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=True)(branch)
    layer = Add()([branch, layer])

    encoder_blocks.append(layer)

    for _ in range(depth - 1):
        filters *= 2
        layer = ResBlock(filters, strides=2)(layer)

        encoder_blocks.append(layer)

    # # 在此处加入 CBAM 模块到每个编码器块的末尾
    # for i, block in enumerate(encoder_blocks):
    #     encoder_blocks[i] = CBAM(block)

    # BRIDGE
    filters *= 2
    layer = ResBlock(filters, strides=2)(layer)

    # DECODER
    for i in range(1, depth + 1):
        filters //= 2
        skip_block_connection = encoder_blocks[-i]

        layer = UpSampling2D()(layer)
        layer = Concatenate()([layer, skip_block_connection])
        layer = ResBlock(filters, strides=1)(layer)
        # 加入 CBAM 模块
        # layer = CBAM(layer)

    layer = Conv2D(filters=classes, kernel_size=1, strides=1, padding="same")(layer)

    if classes == 1:
        layer = Activation('sigmoid', name='Classification')(layer)
    else:
        layer = Activation('softmax', name='Classification')(layer)

    output = layer

    return Model(inputs, output)
