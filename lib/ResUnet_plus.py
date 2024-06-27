import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Concatenate, Activation
from tensorflow.keras.models import Model
from .ResUnet_plus_blocks import *
from .PAAB import PAAB

class ResUnetPlusPlus(tf.keras.Model):
    def __init__(self, channel, filters=[64, 128, 256, 512, 1024]):
        super(ResUnetPlusPlus, self).__init__()

        self.input_layer = tf.keras.Sequential([
            Conv2D(filters[0], kernel_size=3, padding='same', input_shape=(None, None, channel)),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters[0], kernel_size=3, padding='same'),
        ])
        self.input_skip = Conv2D(filters[0], kernel_size=3, padding='same')

        # Assuming Squeeze_Excite_Block, ResidualConv, ASPP, AttentionBlock, and Upsample_ are already implemented for TensorFlow
        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])

        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, padding='same')

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])

        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, padding='same')

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])

        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, padding='same')

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, padding='same')

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, padding='same')

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, padding='same')

        self.aspp_out = ASPP(filters[1], filters[0])

        self.output_layer = tf.keras.Sequential([
            Conv2D(1, kernel_size=1),
            Activation('sigmoid')
        ])

    def call(self, inputs):
        x = self.input_layer(inputs) + self.input_skip(inputs)

        x1 = self.squeeze_excite1(x)
        x2 = self.residual_conv1(x1)

        x3 = self.squeeze_excite2(x2)
        x4 = self.residual_conv2(x3)

        x5 = self.squeeze_excite3(x4)
        x6 = self.residual_conv3(x5)

        x7 = self.aspp_bridge(x6)

        x8 = self.attn1(x4, x7)
        x8 = self.upsample1(x8)
        x8 = Concatenate()([x8, x4])
        x8 = self.up_residual_conv1(x8)

        x9 = self.attn2(x2, x8)
        x9 = self.upsample2(x9)
        x9 = Concatenate()([x9, x2])
        x9 = self.up_residual_conv2(x9)

        x10 = self.attn3(x, x9)
        x10 = self.upsample3(x10)
        x10 = Concatenate()([x10, x])
        x10 = self.up_residual_conv3(x10)

        x11 = self.aspp_out(x10)
        return self.output_layer(x11)