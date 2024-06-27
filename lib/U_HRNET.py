from .U_HRNET_utils import *
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

class UHRNet(Model):
    def __init__(self,
                 in_channels=7,
                 nclasses=1,  # 添加nclasses参数
                 pretrained=None,
                 stage1_num_modules=1,
                 stage1_num_blocks=(4, ),  # 修改为元组
                 stage1_num_channels=(64,),
                 stage2_num_modules=1,
                 stage2_num_blocks=(4, 4),
                 stage2_num_channels=(18, 36),
                 stage3_num_modules=5,
                 stage3_num_blocks=(4, 4),
                 stage3_num_channels=(36, 72),
                 stage4_num_modules=2,
                 stage4_num_blocks=(4, 4),
                 stage4_num_channels=(72, 144),
                 stage5_num_modules=2,
                 stage5_num_blocks=(4, 4),
                 stage5_num_channels=(144, 288),
                 stage6_num_modules=1,
                 stage6_num_blocks=(4, 4),
                 stage6_num_channels=(72, 144),
                 stage7_num_modules=1,
                 stage7_num_blocks=(4, 4),
                 stage7_num_channels=(36, 72),
                 stage8_num_modules=1,
                 stage8_num_blocks=(4, 4),
                 stage8_num_channels=(18, 36),
                 stage9_num_modules=1,
                 stage9_num_blocks=(4,),  # 修改为元组
                 stage9_num_channels=(18,),
                 has_se=False,
                 align_corners=False):
        super(UHRNet, self).__init__()
        self.pretrained = pretrained
        self.has_se = has_se
        self.align_corners = align_corners
        self.feat_channels = [
            sum([
                stage5_num_channels[-1], stage6_num_channels[-1],
                stage7_num_channels[-1], stage8_num_channels[-1],
                stage9_num_channels[-1]
            ]) // 2
        ]

        cur_stride = 1
        # stem net
        self.conv_layer1_1 = ConvBNReLU(
            out_channels=64,
            kernel_size=3,
            stride=2)
        cur_stride *= 2

        self.conv_layer1_2 = ConvBNReLU(
            out_channels=64,
            kernel_size=3,
            stride=2)
        cur_stride *= 2

        self.la1 = Layer1(
            num_channels=64,
            num_blocks=stage1_num_blocks[0],
            num_filters=stage1_num_channels[0],
            has_se=has_se,
            name="layer2")

        self.tr1 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage1_num_channels[0] * 4,
            stride_cur=[
                cur_stride * (2**i) for i in range(len(stage2_num_channels))
            ],
            out_channels=stage2_num_channels,
            align_corners=self.align_corners,
            name="tr1")
        self.st2 = Stage(
            num_channels=stage2_num_channels,
            num_modules=stage2_num_modules,
            num_blocks=stage2_num_blocks,
            num_filters=stage2_num_channels,
            has_se=self.has_se,
            name="st2",
            align_corners=align_corners)
        cur_stride *= 2

        self.tr2 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage2_num_channels[-1],
            stride_cur=[
                cur_stride * (2**i) for i in range(len(stage3_num_channels))
            ],
            out_channels=stage3_num_channels,
            align_corners=self.align_corners,
            name="tr2")
        self.st3 = Stage(
            num_channels=stage3_num_channels,
            num_modules=stage3_num_modules,
            num_blocks=stage3_num_blocks,
            num_filters=stage3_num_channels,
            has_se=self.has_se,
            name="st3",
            align_corners=align_corners)
        cur_stride *= 2

        self.tr3 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage3_num_channels[-1],
            stride_cur=[
                cur_stride * (2**i) for i in range(len(stage4_num_channels))
            ],
            out_channels=stage4_num_channels,
            align_corners=self.align_corners,
            name="tr3")
        self.st4 = Stage(
            num_channels=stage4_num_channels,
            num_modules=stage4_num_modules,
            num_blocks=stage4_num_blocks,
            num_filters=stage4_num_channels,
            has_se=self.has_se,
            name="st4",
            align_corners=align_corners)
        cur_stride *= 2

        self.tr4 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage4_num_channels[-1],
            stride_cur=[
                cur_stride * (2**i) for i in range(len(stage5_num_channels))
            ],
            out_channels=stage5_num_channels,
            align_corners=self.align_corners,
            name="tr4")
        self.st5 = Stage(
            num_channels=stage5_num_channels,
            num_modules=stage5_num_modules,
            num_blocks=stage5_num_blocks,
            num_filters=stage5_num_channels,
            has_se=self.has_se,
            name="st5",
            align_corners=align_corners)

        self.tr5 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage5_num_channels[0],
            stride_cur=[
                cur_stride // (2**(len(stage6_num_channels) - i - 1))
                for i in range(len(stage6_num_channels))
            ],
            out_channels=stage6_num_channels,
            align_corners=self.align_corners,
            name="tr5")
        self.st6 = Stage(
            num_channels=stage6_num_channels,
            num_modules=stage6_num_modules,
            num_blocks=stage6_num_blocks,
            num_filters=stage6_num_channels,
            has_se=self.has_se,
            name="st6",
            align_corners=align_corners)
        cur_stride = cur_stride // 2

        self.tr6 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage6_num_channels[0],
            stride_cur=[
                cur_stride // (2**(len(stage7_num_channels) - i - 1))
                for i in range(len(stage7_num_channels))
            ],
            out_channels=stage7_num_channels,
            align_corners=self.align_corners,
            name="tr6")
        self.st7 = Stage(
            num_channels=stage7_num_channels,
            num_modules=stage7_num_modules,
            num_blocks=stage7_num_blocks,
            num_filters=stage7_num_channels,
            has_se=self.has_se,
            name="st7",
            align_corners=align_corners)
        cur_stride = cur_stride // 2

        self.tr7 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage7_num_channels[0],
            stride_cur=[
                cur_stride // (2**(len(stage8_num_channels) - i - 1))
                for i in range(len(stage8_num_channels))
            ],
            out_channels=stage8_num_channels,
            align_corners=self.align_corners,
            name="tr7")
        self.st8 = Stage(
            num_channels=stage8_num_channels,
            num_modules=stage8_num_modules,
            num_blocks=stage8_num_blocks,
            num_filters=stage8_num_channels,
            has_se=self.has_se,
            name="st8",
            align_corners=align_corners)
        cur_stride = cur_stride // 2

        self.tr8 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage8_num_channels[0],
            stride_cur=[
                cur_stride // (2**(len(stage9_num_channels) - i - 1))
                for i in range(len(stage9_num_channels))
            ],
            out_channels=stage9_num_channels,
            align_corners=self.align_corners,
            name="tr8")
        self.st9 = Stage(
            num_channels=stage9_num_channels,
            num_modules=stage9_num_modules,
            num_blocks=stage9_num_blocks,
            num_filters=stage9_num_channels,
            has_se=self.has_se,
            name="st9",
            align_corners=align_corners)

        self.last_layer = Sequential([
            ConvBNReLU(
                out_channels=self.feat_channels[0],
                kernel_size=1,
                stride=1),
            layers.Conv2D(
                filters=nclasses,  # 使用动态过滤器数量
                kernel_size=1,
                strides=1,
                padding='valid',
                activation='sigmoid' if nclasses == 1 else None)  # 动态选择激活函数
        ])
        self.init_weight()

    def _concat(self, x1, x2):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)

        # 调整 x1 的大小，使其与 x2 的形状匹配
        if x1_shape[1] != x2_shape[1] or x1_shape[2] != x2_shape[2]:
            x1 = tf.image.resize(x1, size=[x2_shape[1], x2_shape[2]], method='bilinear')
        
        x1 = tf.nn.avg_pool3d(
            tf.expand_dims(x1, axis=-1), ksize=(1, 1, 2), strides=(1, 1, 2), padding='VALID')
        x2 = tf.nn.avg_pool3d(
            tf.expand_dims(x2, axis=-1), ksize=(1, 1, 2), strides=(1, 1, 2), padding='VALID')
        
        x1 = tf.squeeze(x1, axis=-1)
        x2 = tf.squeeze(x2, axis=-1)
        
        return tf.concat([x1, x2], axis=-1)
    
    def call(self, x0):
        # conv1 = self.conv_layer1_1(x0)
        # conv2 = self.conv_layer1_2(conv1)

        la1 = self.la1(x0)

        tr1 = self.tr1(la1)
        st2 = self.st2(tr1)
        skip21 = st2[0]

        tr2 = self.tr2(st2[-1])
        st3 = self.st3(tr2)
        skip31 = st3[0]

        tr3 = self.tr3(st3[-1])
        st4 = self.st4(tr3)
        skip41 = st4[0]

        tr4 = self.tr4(st4[-1])
        st5 = self.st5(tr4)
        x5 = st5[-1]

        shape41 = tf.shape(skip41)[1:3]  # 提取高度和宽度
        tr5 = self.tr5(st5[0], shape=shape41)  # 调整大小
        tr5[0] = self._concat(tr5[0], skip41)
        st6 = self.st6(tr5)
        x4 = st6[-1]

        shape31 = tf.shape(skip31)[1:3]  # 提取高度和宽度
        tr6 = self.tr6(st6[0], shape=shape31)  # 调整大小
        tr6[0] = self._concat(tr6[0], skip31)
        st7 = self.st7(tr6)
        x3 = st7[-1]

        shape21 = tf.shape(skip21)[1:3]  # 提取高度和宽度
        tr7 = self.tr7(st7[0], shape=shape21)  # 调整大小
        tr7[0] = self._concat(tr7[0], skip21)
        st8 = self.st8(tr7)
        x2 = st8[-1]

        tr8 = self.tr8(st8[0])
        st9 = self.st9(tr8)
        x1 = st9[-1]

        x = [x1, x2, x3, x4, x5]
        for i in range(len(x)):
            x[i] = tf.nn.avg_pool3d(
                tf.expand_dims(x[i], axis=-1), ksize=(1, 1, 2), strides=(1, 1, 2), padding='VALID')
            x[i] = tf.squeeze(x[i], axis=-1)

        # upsampling
        shape_x0 = tf.shape(x[0])[1:3]
        for i in range(1, len(x)):
            x[i] = tf.image.resize(
                x[i],
                size=shape_x0,
                method='bilinear',
                # align_corners=self.align_corners
                )
        x = tf.concat(x, axis=-1)

        output = self.last_layer(x)

        return output
        # return tf.keras.Model(inputs=x0, outputs=output, name='UHRNet')

    def init_weight(self):
        for layer in self.layers:
            if isinstance(layer, layers.Conv2D):
                layer.kernel_initializer = tf.random_normal_initializer(stddev=0.001)
            elif isinstance(layer, layers.BatchNormalization):
                layer.gamma_initializer = tf.constant_initializer(1.0)
                layer.beta_initializer = tf.constant_initializer(0.0)
        if self.pretrained is not None:
            self.load_pretrained_model()

    def load_pretrained_model(self):
        # Implement loading pretrained model if needed
        pass



def UHRNet_W18_Small(**kwargs):
    model = UHRNet(
        stage1_num_modules=1,
        stage1_num_blocks=(2,),
        stage1_num_channels=(64,),
        stage2_num_modules=1,
        stage2_num_blocks=(2, 2),
        stage2_num_channels=(18, 36),
        stage3_num_modules=2,
        stage3_num_blocks=(2, 2),
        stage3_num_channels=(36, 72),
        stage4_num_modules=2,
        stage4_num_blocks=(2, 2),
        stage4_num_channels=(72, 144),
        stage5_num_modules=2,
        stage5_num_blocks=(2, 2),
        stage5_num_channels=(144, 288),
        stage6_num_modules=1,
        stage6_num_blocks=(2, 2),
        stage6_num_channels=(72, 144),
        stage7_num_modules=1,
        stage7_num_blocks=(2, 2),
        stage7_num_channels=(36, 72),
        stage8_num_modules=1,
        stage8_num_blocks=(2, 2),
        stage8_num_channels=(18, 36),
        stage9_num_modules=1,
        stage9_num_blocks=(2,),
        stage9_num_channels=(18,),
        **kwargs
    )
    return model


def UHRNet_W18(**kwargs):
    model = UHRNet(
        stage1_num_modules=1,
        stage1_num_blocks=(4,),
        stage1_num_channels=(64,),
        stage2_num_modules=1,
        stage2_num_blocks=(4, 4),
        stage2_num_channels=(18, 36),
        stage3_num_modules=5,
        stage3_num_blocks=(4, 4),
        stage3_num_channels=(36, 72),
        stage4_num_modules=2,
        stage4_num_blocks=(4, 4),
        stage4_num_channels=(72, 144),
        stage5_num_modules=2,
        stage5_num_blocks=(4, 4),
        stage5_num_channels=(144, 288),
        stage6_num_modules=1,
        stage6_num_blocks=(4, 4),
        stage6_num_channels=(72, 144),
        stage7_num_modules=1,
        stage7_num_blocks=(4, 4),
        stage7_num_channels=(36, 72),
        stage8_num_modules=1,
        stage8_num_blocks=(4, 4),
        stage8_num_channels=(18, 36),
        stage9_num_modules=1,
        stage9_num_blocks=(4,),
        stage9_num_channels=(18,),
        **kwargs
    )
    return model

def UHRNet_W48(**kwargs):
    model = UHRNet(
        stage1_num_modules=1,
        stage1_num_blocks=(4,),
        stage1_num_channels=(64,),
        stage2_num_modules=1,
        stage2_num_blocks=(4, 4),
        stage2_num_channels=(48, 96),
        stage3_num_modules=5,
        stage3_num_blocks=(4, 4),
        stage3_num_channels=(96, 192),
        stage4_num_modules=2,
        stage4_num_blocks=(4, 4),
        stage4_num_channels=(192, 384),
        stage5_num_modules=2,
        stage5_num_blocks=(4, 4),
        stage5_num_channels=(384, 768),
        stage6_num_modules=1,
        stage6_num_blocks=(4, 4),
        stage6_num_channels=(192, 384),
        stage7_num_modules=1,
        stage7_num_blocks=(4, 4),
        stage7_num_channels=(96, 192),
        stage8_num_modules=1,
        stage8_num_blocks=(4, 4),
        stage8_num_channels=(48, 96),
        stage9_num_modules=1,
        stage9_num_blocks=(4,),
        stage9_num_channels=(48,),
        **kwargs
    )
    return model