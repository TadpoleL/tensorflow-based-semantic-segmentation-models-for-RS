import tensorflow as tf
import math

class ConvBNReLU(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size, stride=1, padding='same', name=None, **kwargs):
        super(ConvBNReLU, self).__init__(name=name)

        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride, 
            padding=padding,
            **kwargs)

        self.batch_norm = tf.keras.layers.BatchNormalization()

        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.batch_norm(x, training=training)
        x = self.relu(x)
        return x


class ConvBN(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size, stride=1, padding='same', name=None, **kwargs):
        super(ConvBN, self).__init__(name=name)

        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride, 
            padding=padding,
            **kwargs)

        self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.batch_norm(x, training=training)
        return x


class SELayer(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_filters, reduction_ratio, name=None):
        super(SELayer, self).__init__()
        
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        med_ch = num_channels // reduction_ratio

        # Initialize the squeeze layer
        self.squeeze = tf.keras.layers.Dense(
            units=med_ch,
            activation='relu',
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-1.0 / math.sqrt(num_channels),
                maxval=1.0 / math.sqrt(num_channels)
            )
        )

        # Initialize the excitation layer
        self.excitation = tf.keras.layers.Dense(
            units=num_filters,
            activation='sigmoid',
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-1.0 / math.sqrt(med_ch),
                maxval=1.0 / math.sqrt(med_ch)
            )
        )

    def call(self, inputs):
        pool = self.global_avg_pool(inputs)
        squeeze = self.squeeze(pool)
        excitation = self.excitation(squeeze)
        excitation = tf.reshape(excitation, [-1, 1, 1, excitation.shape[-1]])
        out = inputs * excitation
        return out

def avg_pool3d(x, kernel_size, stride, padding='VALID'):
    # TensorFlow expects kernel_size and stride to be of length 5 (batch, depth, height, width, channels)
    kernel_size = [1] + list(kernel_size) + [1]
    stride = [1] + list(stride) + [1]
    return tf.nn.avg_pool3d(x, ksize=kernel_size, strides=stride, padding=padding)

class FuseLayers(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, multi_scale_output=True, name=None, align_corners=False):
        super(FuseLayers, self).__init__()
        self._actual_ch = len(in_channels) if multi_scale_output else 1
        self._in_channels = in_channels
        self.align_corners = align_corners

        self.residual_func_list = []
        for i in range(self._actual_ch):
            for j in range(len(in_channels)):
                if j > i:
                    residual_func = ConvBN(
                        # in_channels=in_channels[j],
                        out_channels=out_channels[i],
                        kernel_size=1,
                        padding='same',
                        use_bias=False)
                    self.residual_func_list.append(residual_func)
                elif j < i:
                    pre_num_filters = in_channels[j]
                    for k in range(i - j):
                        if k == i - j - 1:
                            residual_func = ConvBN(
                                # in_channels=pre_num_filters,
                                out_channels=out_channels[i],
                                kernel_size=3,
                                stride=2,
                                padding='same',
                                use_bias=False)
                            pre_num_filters = out_channels[i]
                        else:
                            residual_func = ConvBNReLU(
                                # in_channels=pre_num_filters,
                                out_channels=out_channels[j],
                                kernel_size=3,
                                stride=2,
                                padding='same',
                                use_bias=False)
                            pre_num_filters = out_channels[j]
                        self.residual_func_list.append(residual_func)

        if len(self.residual_func_list) == 0:
            self.residual_func_list.append(tf.keras.layers.Lambda(lambda x: x))  # for flops calculation

    def call(self, inputs, training=False):
        outs = []
        residual_func_idx = 0
        for i in range(self._actual_ch):
            residual = inputs[i]
            residual_shape = tf.shape(residual)[1:3]

            for j in range(len(self._in_channels)):
                if j > i:
                    y = self.residual_func_list[residual_func_idx](inputs[j], training=training)
                    residual_func_idx += 1

                    y = tf.image.resize(y, residual_shape, method='bilinear')
                    residual = residual + y
                elif j < i:
                    y = inputs[j]
                    for k in range(i - j):
                        y = self.residual_func_list[residual_func_idx](y, training=training)
                        residual_func_idx += 1

                    residual = residual + y

            residual = tf.nn.relu(residual)
            outs.append(residual)

        return outs


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_filters, stride=1, has_se=False, downsample=False, name=None):
        super(BasicBlock, self).__init__()

        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = ConvBNReLU(
            out_channels=num_filters,
            kernel_size=3,
            stride=stride,
            padding='same')
        self.conv2 = ConvBN(
            out_channels=num_filters,
            kernel_size=3,
            padding='same')

        if self.downsample:
            self.conv_down = ConvBNReLU(
                out_channels=num_filters,
                kernel_size=1,
                padding='same')

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters,
                num_filters=num_filters,
                reduction_ratio=16,
                name=name + '_fc')

    def call(self, inputs, training=False):
        residual = inputs
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)

        if self.downsample:
            residual = self.conv_down(inputs, training=training)

        if self.has_se:
            x = self.se(x)

        x += residual
        x = tf.nn.relu(x)
        return x


class Branches(tf.keras.layers.Layer):
    def __init__(self, num_blocks, in_channels, out_channels, has_se=False, name=None):
        super(Branches, self).__init__()

        self.basic_block_list = []

        for i in range(len(out_channels)):
            branch = []
            for j in range(num_blocks[i]):
                in_ch = in_channels[i] if j == 0 else out_channels[i]
                basic_block = BasicBlock(
                    num_channels=in_ch,
                    num_filters=out_channels[i],
                    has_se=has_se,
                    name=name + '_branch_layer_' + str(i + 1) + '_' + str(j + 1))
                branch.append(basic_block)
            self.basic_block_list.append(branch)

    def call(self, inputs, training=False):
        outs = []
        for idx, input in enumerate(inputs):
            x = input
            for basic_block in self.basic_block_list[idx]:
                x = basic_block(x, training=training)
            outs.append(x)
        return outs


class BottleneckBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_filters, has_se, stride=1, downsample=False, name=None):
        super(BottleneckBlock, self).__init__(name=name)
        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = ConvBNReLU(out_channels=num_filters, kernel_size=1, padding='same', name=name+'_conv1')
        self.conv2 = ConvBNReLU(out_channels=num_filters, kernel_size=3, stride=stride, padding='same', name=name+'_conv2')
        self.conv3 = ConvBN(out_channels=num_filters * 4, kernel_size=1, padding='same', name=name+'_conv3')

        if self.downsample:
            self.conv_down = ConvBN(out_channels=num_filters * 4, kernel_size=1, padding='same', name=name+'_conv_down')

        if self.has_se:
            self.se = SELayer(num_channels=num_filters * 4, reduction_ratio=16, name=name+'_se')

    def call(self, inputs, training=False):
        residual = inputs
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)

        if self.downsample:
            residual = self.conv_down(inputs, training=training)

        if self.has_se:
            x = self.se(x)

        x = tf.keras.layers.add([x, residual])
        x = tf.keras.layers.ReLU()(x)
        return x

class HighResolutionModule(tf.keras.layers.Layer):
    def __init__(self,
                 num_channels,
                 num_blocks,
                 num_filters,
                 has_se=False,
                 multi_scale_output=True,
                 name=None,
                 align_corners=False):
        super(HighResolutionModule, self).__init__()

        self.branches_func = Branches(
            num_blocks=num_blocks,
            in_channels=num_channels,
            out_channels=num_filters,
            has_se=has_se,
            name=name)

        self.fuse_func = FuseLayers(
            in_channels=num_filters,
            out_channels=num_filters,
            multi_scale_output=multi_scale_output,
            name=name,
            align_corners=align_corners)

    def call(self, x, training=False):
        out = self.branches_func(x, training=training)
        out = self.fuse_func(out, training=training)
        return out


class Stage(tf.keras.layers.Layer):
    def __init__(self,
                 num_channels,
                 num_modules,
                 num_blocks,
                 num_filters,
                 has_se=False,
                 multi_scale_output=True,
                 name=None,
                 align_corners=False):
        super(Stage, self).__init__()

        self._num_modules = num_modules

        self.stage_func_list = []
        for i in range(num_modules):
            if i == num_modules - 1 and not multi_scale_output:
                stage_func = HighResolutionModule(
                    num_channels=num_channels,
                    num_blocks=num_blocks,
                    num_filters=num_filters,
                    has_se=has_se,
                    multi_scale_output=False,
                    name=name + '_' + str(i + 1),
                    align_corners=align_corners)
            else:
                stage_func = HighResolutionModule(
                    num_channels=num_channels,
                    num_blocks=num_blocks,
                    num_filters=num_filters,
                    has_se=has_se,
                    name=name + '_' + str(i + 1),
                    align_corners=align_corners)

            self.stage_func_list.append(stage_func)

    def call(self, x, training=False):
        out = x
        for idx in range(self._num_modules):
            out = self.stage_func_list[idx](out, training=training)
        return out


class Layer1(tf.keras.layers.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 num_blocks,
                 has_se=False,
                 name=None):
        super(Layer1, self).__init__()

        self.bottleneck_block_list = []

        for i in range(num_blocks):
            bottleneck_block = BottleneckBlock(
                num_channels=num_channels if i == 0 else num_filters * 4,
                num_filters=num_filters,
                has_se=has_se,
                stride=1,
                downsample=True if i == 0 else False,
                name=name + '_' + str(i + 1))
            self.bottleneck_block_list.append(bottleneck_block)

    def call(self, x, training=False):
        conv = x
        for block_func in self.bottleneck_block_list:
            conv = block_func(conv, training=training)
        return conv

class TransitionLayer(tf.keras.layers.Layer):
    def __init__(self, stride_pre, in_channel, stride_cur, out_channels, align_corners=False, name=None):
        super(TransitionLayer, self).__init__(name=name)
        self.align_corners = align_corners
        self.conv_bn_func_list = []

        for i in range(len(out_channels)):
            residual = None
            if stride_cur[i] == stride_pre:
                if in_channel != out_channels[i]:
                    residual = ConvBNReLU(
                        out_channels=out_channels[i],
                        kernel_size=3,
                        padding='same',
                        use_bias=False)
            elif stride_cur[i] > stride_pre:
                residual = ConvBNReLU(
                    out_channels=out_channels[i],
                    kernel_size=3,
                    stride=2,
                    padding='same',
                    use_bias=False)
            else:  # stride_cur[i] < stride_pre
                residual = ConvBNReLU(
                    out_channels=out_channels[i],
                    kernel_size=1,
                    stride=1,
                    padding='same',
                    use_bias=False)
            self.conv_bn_func_list.append(residual)

    def call(self, x, shape=None, training=False):
        outs = []
        for conv_bn_func in self.conv_bn_func_list:
            if conv_bn_func is None:
                outs.append(x)
            else:
                out = conv_bn_func(x, training=training)
                if shape is not None:
                    out = tf.image.resize(out, size=shape, method='bilinear')
                outs.append(out)
        return outs
