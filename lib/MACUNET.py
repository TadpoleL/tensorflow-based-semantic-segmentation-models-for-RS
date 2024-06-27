import tensorflow as tf
from tensorflow.keras import layers, models

class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, in_planes, out_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.conv = layers.Conv2D(out_planes, 1, use_bias=False)
        self.avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.max_pool = layers.GlobalMaxPooling2D(keepdims=True)

        self.fc11 = layers.Conv2D(out_planes // ratio, 1, use_bias=False)
        self.fc12 = layers.Conv2D(out_planes, 1, use_bias=False)

        self.fc21 = layers.Conv2D(out_planes // ratio, 1, use_bias=False)
        self.fc22 = layers.Conv2D(out_planes, 1, use_bias=False)
        self.relu1 = layers.ReLU()

        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x):
        x = self.conv(x)
        avg_out = self.fc12(self.relu1(self.fc11(self.avg_pool(x))))
        max_out = self.fc22(self.relu1(self.fc21(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)

def conv3otherRelu(in_planes, out_planes, kernel_size=3, stride=1, padding='same'):
    return models.Sequential([
        layers.Conv2D(out_planes, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=True),
        layers.LeakyReLU()
    ])

class ACBlock(tf.keras.layers.Layer):
    def __init__(self, in_planes, out_planes):
        super(ACBlock, self).__init__()
        self.squre = layers.Conv2D(out_planes, kernel_size=3, padding='same', strides=1)
        self.cross_ver = layers.Conv2D(out_planes, kernel_size=(1, 3), padding='same', strides=1)
        self.cross_hor = layers.Conv2D(out_planes, kernel_size=(3, 1), padding='same', strides=1)
        self.bn = layers.BatchNormalization()
        self.ReLU = layers.ReLU()

    def call(self, x):
        x1 = self.squre(x)
        x2 = self.cross_ver(x)
        x3 = self.cross_hor(x)
        return self.ReLU(self.bn(x1 + x2 + x3))

class MACUNet(tf.keras.Model):
    def __init__(self, band_num, class_num):
        super(MACUNet, self).__init__()
        self.band_num = band_num
        self.class_num = class_num

        # channels = [16, 32, 64, 128, 256, 512]
        channels = [32, 64, 128, 256, 512, 1024]
        
        self.conv1 = models.Sequential([
            ACBlock(band_num, channels[0]),
            ACBlock(channels[0], channels[0])
        ])
        self.conv12 = models.Sequential([
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            ACBlock(channels[0], channels[1])
        ])
        self.conv13 = models.Sequential([
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            ACBlock(channels[1], channels[2])
        ])
        self.conv14 = models.Sequential([
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            ACBlock(channels[2], channels[3])
        ])

        self.conv2 = models.Sequential([
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            ACBlock(channels[0], channels[1]),
            ACBlock(channels[1], channels[1])
        ])
        self.conv23 = models.Sequential([
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            ACBlock(channels[1], channels[2])
        ])
        self.conv24 = models.Sequential([
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            ACBlock(channels[2], channels[3])
        ])

        self.conv3 = models.Sequential([
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            ACBlock(channels[1], channels[2]),
            ACBlock(channels[2], channels[2]),
            ACBlock(channels[2], channels[2])
        ])
        self.conv34 = models.Sequential([
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            ACBlock(channels[2], channels[3])
        ])

        self.conv4 = models.Sequential([
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            ACBlock(channels[2], channels[3]),
            ACBlock(channels[3], channels[3]),
            ACBlock(channels[3], channels[3])
        ])

        self.conv5 = models.Sequential([
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            ACBlock(channels[3], channels[4]),
            ACBlock(channels[4], channels[4]),
            ACBlock(channels[4], channels[4])
        ])

        self.skblock4 = ChannelAttention(channels[3]*5, channels[3]*2, 16)
        self.skblock3 = ChannelAttention(channels[2]*5, channels[2]*2, 16)
        self.skblock2 = ChannelAttention(channels[1]*5, channels[1]*2, 16)
        self.skblock1 = ChannelAttention(channels[0]*5, channels[0]*2, 16)

        self.deconv4 = layers.Conv2DTranspose(channels[3], kernel_size=(2, 2), strides=(2, 2))
        self.deconv43 = layers.Conv2DTranspose(channels[2], kernel_size=(2, 2), strides=(2, 2))
        self.deconv42 = layers.Conv2DTranspose(channels[1], kernel_size=(2, 2), strides=(2, 2))
        self.deconv41 = layers.Conv2DTranspose(channels[0], kernel_size=(2, 2), strides=(2, 2))

        self.conv6 = models.Sequential([
            ACBlock(channels[4], channels[3]),
            ACBlock(channels[3], channels[3])
        ])

        self.deconv3 = layers.Conv2DTranspose(channels[2], kernel_size=(2, 2), strides=(2, 2))
        self.deconv32 = layers.Conv2DTranspose(channels[1], kernel_size=(2, 2), strides=(2, 2))
        self.deconv31 = layers.Conv2DTranspose(channels[0], kernel_size=(2, 2), strides=(2, 2))
        self.conv7 = models.Sequential([
            ACBlock(channels[3], channels[2]),
            ACBlock(channels[2], channels[2])
        ])

        self.deconv2 = layers.Conv2DTranspose(channels[1], kernel_size=(2, 2), strides=(2, 2))
        self.deconv21 = layers.Conv2DTranspose(channels[0], kernel_size=(2, 2), strides=(2, 2))
        self.conv8 = models.Sequential([
            ACBlock(channels[2], channels[1]),
            ACBlock(channels[1], channels[1])
        ])

        self.deconv1 = layers.Conv2DTranspose(channels[0], kernel_size=(2, 2), strides=(2, 2))
        self.conv9 = models.Sequential([
            ACBlock(channels[1], channels[0]),
            ACBlock(channels[0], channels[0])
        ])

        self.conv10 = layers.Conv2D(class_num, kernel_size=1, strides=1)

        if class_num == 1:
            self.final_activation = layers.Activation('sigmoid')
        else:
            self.final_activation = layers.Activation('softmax')

    def call(self, x):
        conv1 = self.conv1(x)
        conv12 = self.conv12(conv1)
        conv13 = self.conv13(conv12)
        conv14 = self.conv14(conv13)

        conv2 = self.conv2(conv1)
        conv23 = self.conv23(conv2)
        conv24 = self.conv24(conv23)

        conv3 = self.conv3(conv2)
        conv34 = self.conv34(conv3)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv4 = self.deconv4(conv5)
        deconv43 = self.deconv43(deconv4)
        deconv42 = self.deconv42(deconv43)
        deconv41 = self.deconv41(deconv42)

        conv6 = tf.concat([deconv4, conv4, conv34, conv24, conv14], axis=-1)
        conv6 = self.skblock4(conv6)
        conv6 = self.conv6(conv6)

        deconv3 = self.deconv3(conv6)
        deconv32 = self.deconv32(deconv3)
        deconv31 = self.deconv31(deconv32)

        conv7 = tf.concat([deconv3, deconv43, conv3, conv23, conv13], axis=-1)
        conv7 = self.skblock3(conv7)
        conv7 = self.conv7(conv7)

        deconv2 = self.deconv2(conv7)
        deconv21 = self.deconv21(deconv2)

        conv8 = tf.concat([deconv2, deconv42, deconv32, conv2, conv12], axis=-1)
        conv8 = self.skblock2(conv8)
        conv8 = self.conv8(conv8)

        deconv1 = self.deconv1(conv8)
        conv9 = tf.concat([deconv1, deconv41, deconv31, deconv21, conv1], axis=-1)
        conv9 = self.skblock1(conv9)
        conv9 = self.conv9(conv9)

        conv10 = self.conv10(conv9)

        output = self.final_activation(conv10)

        return output

# if __name__ == '__main__':
#     num_classes = 10
#     in_batch, inchannel, in_h, in_w = 4, 3, 128, 128
#     x = tf.random.normal([in_batch, inchannel, in_h, in_w])
#     net = MACUNet(inchannel, num_classes)
#     out = net(x)
#     print(out.shape)
