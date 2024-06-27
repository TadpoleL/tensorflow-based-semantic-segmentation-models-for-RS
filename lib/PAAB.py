from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, Concatenate
import tensorflow as tf
import numpy as np

class PAAB(Layer):
    def __init__(self, **kwargs):
        super(PAAB, self).__init__(**kwargs)
        # 1x1卷积生成gamma和beta的空间映射
        self.conv_gamma = Conv2D(filters=1, kernel_size=(1, 1), padding='same')
        self.conv_beta = Conv2D(filters=1, kernel_size=(1, 1), padding='same')

    def call(self, inputs):
        # S是扁平的编码向量，F是卷积特征图
        S, F = inputs

        # tf.print("S shape:", S.shape)
        # tf.print("F shape:", F.shape)

        # # 将S转换为具有空间维度的形状
        # S_expanded = S[:, tf.newaxis, tf.newaxis, :]
        # S_expanded = tf.tile(S_expanded, [1, tf.shape(F)[1], tf.shape(F)[2], 1])

        # 扩展S以匹配F的空间维度
        S_expanded = tf.expand_dims(tf.expand_dims(S, 1), 1)
        S_expanded = tf.tile(S_expanded, [1, tf.shape(F)[1], tf.shape(F)[2], 1])
        # tf.print("S_expanded shape:", S_expanded.shape)

        # 使用1x1卷积生成gamma和beta
        gamma = self.conv_gamma(S_expanded)
        beta = self.conv_beta(S_expanded)

        # 应用gamma和beta对F进行调整
        F_adjusted = gamma * F + beta

        # print("F_adjusted shape:", F_adjusted.shape)

        return F_adjusted

    def compute_output_shape(self, input_shape):
        # PAAB层不改变特征图F的形状
        return input_shape[0], input_shape[1]

# class PAAB(Layer):
#     def __init__(self, **kwargs):
#         super(PAAB, self).__init__(**kwargs)
#         self.conv_gamma = Conv2D(filters=1, kernel_size=(1, 1), padding='same')
#         self.conv_beta = Conv2D(filters=1, kernel_size=(1, 1), padding='same')
#         self.batch_norm = BatchNormalization()  # 添加批量归一化层

#     def call(self, inputs):
#         S, F = inputs

#         # 将S转换为具有空间维度的形状并与F连接
#         S_expanded = S[:, tf.newaxis, tf.newaxis, :]
#         S_expanded = tf.tile(S_expanded, [1, tf.shape(F)[1], tf.shape(F)[2], 1])
#         SF_concat = Concatenate(axis=-1)([S_expanded, F])

#         # 使用1x1卷积生成gamma和beta
#         gamma = self.conv_gamma(SF_concat)
#         beta = self.conv_beta(SF_concat)

#         # 首先对F进行批量归一化
#         F_normalized = self.batch_norm(F)

#         # 应用gamma和beta对批量归一化后的F进行调整
#         F_adjusted = gamma * F_normalized + beta
        
#         # 将原来的F与调整后的F连接，形成最终的输出
#         F_final = tf.concat([F, F_adjusted], axis=-1)

#         return F_final

#     def compute_output_shape(self, input_shape):
#         # 更新compute_output_shape以反映F_final的新维度
#         F_shape = input_shape[1]  # 假设input_shape为[S_shape, F_shape]
#         return (F_shape[0], F_shape[1], F_shape[2], F_shape[3] * 2)  # F的深度变为原来的两倍