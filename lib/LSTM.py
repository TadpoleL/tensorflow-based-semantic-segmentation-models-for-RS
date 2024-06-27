from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from utils.keras_utils import train_model, evaluate_model, set_trainable
from utils.layer_utils import AttentionLSTM

# - To use the MLSTM FCN model : `model = generate_model()`
# - To use the MALSTM FCN model : `model = generate_model_2()`
# - To use the LSTM FCN model : `model = generate_model_3()`
# - To use the ALSTM FCN model : `model = generate_model_4()`

class TimeSeriesModels:
    def __init__(self, max_timesteps, max_nb_variables, nb_classes):
        self.MAX_TIMESTEPS = max_timesteps         #时间序列中的最大时间步数，即时间序列的长度
        self.MAX_NB_VARIABLES = max_nb_variables   #时间序列中的变量（或特征）的最大数量
        self.NB_CLASS = nb_classes
        self.TRAINABLE = True

    def squeeze_excite_block(self, input):
        filters = input._keras_shape[-1]  # channel_axis = -1 for TF
        se = GlobalAveragePooling1D()(input)
        se = Reshape((1, filters))(se)
        se = Dense(filters // 16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        se = multiply([input, se])
        return se

    def MLSTM_FCN(self):
        ip = Input(shape=(self.MAX_NB_VARIABLES, self.MAX_TIMESTEPS))
        x = Masking()(ip)
        x = LSTM(8)(x)
        x = Dropout(0.8)(x)
        y = Permute((2, 1))(ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y)
        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y)
        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = GlobalAveragePooling1D()(y)
        x = concatenate([x, y])
        if self.NB_CLASS == 1:
            out = Dense(self.NB_CLASS, activation='sigmoid')(x)
        else:
            out = Dense(self.NB_CLASS, activation='softmax')(x)
        model = Model(ip, out)
        model.summary()
        return model
    
    def MALSTM_FCN(self):
        ip = Input(shape=(self.MAX_NB_VARIABLES,self. MAX_TIMESTEPS))

        ''' sabsample timesteps to prevent OOM due to Attention LSTM '''
        stride = 2

        x = Permute((2, 1))(ip)
        x = Conv1D(self.MAX_NB_VARIABLES // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
                kernel_initializer='he_uniform')(x) # (None, variables / stride, timesteps)
        x = Permute((2, 1))(x)

        x = Masking()(x)
        x = AttentionLSTM(128)(x)
        x = Dropout(0.8)(x)

        y = Permute((2, 1))(ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y)

        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y)

        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])

        if self.NB_CLASS == 1:
            out = Dense(self.NB_CLASS, activation='sigmoid')(x)
        else:
            out = Dense(self.NB_CLASS, activation='softmax')(x)

        model = Model(ip, out)
        model.summary()

        # add load model code here to fine-tune

        return model
    
    def LSTM_FCN(self):
        ip = Input(shape=(self.MAX_NB_VARIABLES, self.MAX_TIMESTEPS))

        x = Masking()(ip)
        x = LSTM(8)(x)
        x = Dropout(0.8)(x)

        y = Permute((2, 1))(ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        #y = squeeze_excite_block(y)

        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        #y = squeeze_excite_block(y)

        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])

        if self.NB_CLASS == 1:
            out = Dense(self.NB_CLASS, activation='sigmoid')(x)
        else:
            out = Dense(self.NB_CLASS, activation='softmax')(x)

        model = Model(ip, out)
        model.summary()

        # add load model code here to fine-tune

        return model

    def ALSTM_FCN(self):
        ip = Input(shape=(self.MAX_NB_VARIABLES,self. MAX_TIMESTEPS))
        # stride = 3
        #
        # x = Permute((2, 1))(ip)
        # x = Conv1D(MAX_NB_VARIABLES // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
        #            kernel_initializer='he_uniform')(x)  # (None, variables / stride, timesteps)
        # x = Permute((2, 1))(x)

        x = Masking()(ip)
        x = AttentionLSTM(8)(x)
        x = Dropout(0.8)(x)

        y = Permute((2, 1))(ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        #y = squeeze_excite_block(y)

        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        #y = squeeze_excite_block(y)

        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])

        if self.NB_CLASS == 1:
            out = Dense(self.NB_CLASS, activation='sigmoid')(x)
        else:
            out = Dense(self.NB_CLASS, activation='softmax')(x)

        model = Model(ip, out)
        model.summary()

        # add load model code here to fine-tune

        return model