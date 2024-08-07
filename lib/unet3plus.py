"""
UNet3+ base model
"""
import tensorflow as tf
import tensorflow.keras as k
from .unet3plus_utils import conv_block


# def unet3plus(input_shape, output_channels):
def unet3plus(height, width, channel, classes=1, filters_root=64):
    """ UNet3+ base model """
    input_shape = (height, width, channel)
    # filters = [filters_root, filters_root*2, filters_root*4, filters_root*8, filters_root*16]
    filters = [filters_root, filters_root*2, filters_root*4, filters_root*8]
    # filters = [64, 128, 256, 512, 1024]

    input_layer = k.layers.Input(
        shape=input_shape,
        name="input_layer"
    )  # 320*320*3

    """ Encoder"""
    # block 1
    e1 = conv_block(input_layer, filters[0])  # 320*320*64

    # block 2
    e2 = k.layers.MaxPool2D(pool_size=(2, 2))(e1)  # 160*160*64
    e2 = conv_block(e2, filters[1])  # 160*160*128

    # block 3
    e3 = k.layers.MaxPool2D(pool_size=(2, 2))(e2)  # 80*80*128
    e3 = conv_block(e3, filters[2])  # 80*80*256

    # block 4
    e4 = k.layers.MaxPool2D(pool_size=(2, 2))(e3)  # 40*40*256
    e4 = conv_block(e4, filters[3])  # 40*40*512

    # # block 5
    # # bottleneck layer
    # e5 = k.layers.MaxPool2D(pool_size=(2, 2))(e4)  # 20*20*512
    # e5 = conv_block(e5, filters[4])  # 20*20*1024

    """ Decoder """
    cat_channels = filters[0]
    cat_blocks = len(filters)
    upsample_channels = cat_blocks * cat_channels

    # """ d4 """
    # e1_d4 = k.layers.MaxPool2D(pool_size=(8, 8))(e1)  # 320*320*64  --> 40*40*64
    # e1_d4 = conv_block(e1_d4, cat_channels, n=1)  # 320*320*64  --> 40*40*64

    # e2_d4 = k.layers.MaxPool2D(pool_size=(4, 4))(e2)  # 160*160*128 --> 40*40*128
    # e2_d4 = conv_block(e2_d4, cat_channels, n=1)  # 160*160*128 --> 40*40*64

    # e3_d4 = k.layers.MaxPool2D(pool_size=(2, 2))(e3)  # 80*80*256  --> 40*40*256
    # e3_d4 = conv_block(e3_d4, cat_channels, n=1)  # 80*80*256  --> 40*40*64

    # e4_d4 = conv_block(e4, cat_channels, n=1)  # 40*40*512  --> 40*40*64

    # e5_d4 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(e5)  # 80*80*256  --> 40*40*256
    # e5_d4 = conv_block(e5_d4, cat_channels, n=1)  # 20*20*1024  --> 20*20*64

    # d4 = k.layers.concatenate([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4])
    # d4 = conv_block(d4, upsample_channels, n=1)  # 40*40*320  --> 40*40*320

    """ d3 """
    e1_d3 = k.layers.MaxPool2D(pool_size=(4, 4))(e1)  # 320*320*64 --> 80*80*64
    e1_d3 = conv_block(e1_d3, cat_channels, n=1)  # 80*80*64 --> 80*80*64

    e2_d3 = k.layers.MaxPool2D(pool_size=(2, 2))(e2)  # 160*160*256 --> 80*80*256
    e2_d3 = conv_block(e2_d3, cat_channels, n=1)  # 80*80*256 --> 80*80*64

    e3_d3 = conv_block(e3, cat_channels, n=1)  # 80*80*512 --> 80*80*64

    e4_d3 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(e4)  # 40*40*320 --> 80*80*320
    e4_d3 = conv_block(e4_d3, cat_channels, n=1)  # 80*80*320 --> 80*80*64

    # e5_d3 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(e5)  # 20*20*320 --> 80*80*320
    # e5_d3 = conv_block(e5_d3, cat_channels, n=1)  # 80*80*320 --> 80*80*64

    # d3 = k.layers.concatenate([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3])
    d3 = k.layers.concatenate([e1_d3, e2_d3, e3_d3, e4_d3])
    d3 = conv_block(d3, upsample_channels, n=1)  # 80*80*320 --> 80*80*320

    """ d2 """
    e1_d2 = k.layers.MaxPool2D(pool_size=(2, 2))(e1)  # 320*320*64 --> 160*160*64
    e1_d2 = conv_block(e1_d2, cat_channels, n=1)  # 160*160*64 --> 160*160*64

    e2_d2 = conv_block(e2, cat_channels, n=1)  # 160*160*256 --> 160*160*64

    d3_d2 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d3)  # 80*80*320 --> 160*160*320
    d3_d2 = conv_block(d3_d2, cat_channels, n=1)  # 160*160*320 --> 160*160*64

    e4_d2 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(e4)  # 40*40*320 --> 160*160*320
    e4_d2 = conv_block(e4_d2, cat_channels, n=1)  # 160*160*320 --> 160*160*64

    # e5_d2 = k.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(e5)  # 20*20*320 --> 160*160*320
    # e5_d2 = conv_block(e5_d2, cat_channels, n=1)  # 160*160*320 --> 160*160*64

    d2 = k.layers.concatenate([e1_d2, e2_d2, d3_d2, e4_d2])
    d2 = conv_block(d2, upsample_channels, n=1)  # 160*160*320 --> 160*160*320

    """ d1 """
    e1_d1 = conv_block(e1, cat_channels, n=1)  # 320*320*64 --> 320*320*64

    d2_d1 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d2)  # 160*160*320 --> 320*320*320
    d2_d1 = conv_block(d2_d1, cat_channels, n=1)  # 160*160*320 --> 160*160*64

    d3_d1 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d3)  # 80*80*320 --> 320*320*320
    d3_d1 = conv_block(d3_d1, cat_channels, n=1)  # 320*320*320 --> 320*320*64

    e4_d1 = k.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(e4)  # 40*40*320 --> 320*320*320
    e4_d1 = conv_block(e4_d1, cat_channels, n=1)  # 320*320*320 --> 320*320*64

    # e5_d1 = k.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(e5)  # 20*20*320 --> 320*320*320
    # e5_d1 = conv_block(e5_d1, cat_channels, n=1)  # 320*320*320 --> 320*320*64

    d1 = k.layers.concatenate([e1_d1, d2_d1, d3_d1, e4_d1])
    d1 = conv_block(d1, upsample_channels, n=1)  # 320*320*320 --> 320*320*320

    # last layer does not have batchnorm and relu
    d = conv_block(d1, classes, n=1, is_bn=False, is_relu=False)
    if classes == 1:
        output = k.activations.sigmoid(d)
    else:
        output = k.activations.softmax(d)

    output = k.activations.softmax(d)

    return tf.keras.Model(inputs=input_layer, outputs=output, name='UNet_3Plus')


def tiny_unet3plus(input_shape, output_channels, training):
    """ Sample model only for testing during development """
    filters = [64, 128, 256, 512, 1024]

    input_layer = k.layers.Input(shape=input_shape, name="input_layer")  # 320*320*3

    """ Encoder"""
    # block 1
    e1 = conv_block(input_layer, filters[0] // 2)  # 320*320*64
    e1 = conv_block(e1, filters[0] // 2)  # 320*320*64

    # last layer does not have batch norm and relu
    d = conv_block(e1, output_channels, n=1, is_bn=False, is_relu=False)
    output = k.activations.softmax(d, )

    if training:
        e2 = conv_block(e1, filters[0] // 2)  # 320*320*64
        d2 = conv_block(e2, output_channels, n=1, is_bn=False, is_relu=False)
        output2 = k.activations.softmax(d2)
        return tf.keras.Model(inputs=input_layer, outputs=[output, output2], name='UNet3Plus')
    else:
        return tf.keras.Model(inputs=input_layer, outputs=[output], name='UNet3Plus')


if __name__ == "__main__":
    """## Model Compilation"""
    INPUT_SHAPE = [320, 320, 1]
    OUTPUT_CHANNELS = 1

    unet_3P = unet3plus(INPUT_SHAPE, OUTPUT_CHANNELS)
    unet_3P.summary()

    # tf.keras.utils.plot_model(unet_3P, show_layer_names=True, show_shapes=True)

    # unet_3P.save("unet_3P.hdf5")
