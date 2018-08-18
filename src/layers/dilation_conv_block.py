from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, BatchNormalization


class DilationConvBlock():
    def __init__(self, filters, dilation_rate):
        self.__filters = filters
        self.__dilation_rate = dilation_rate


    def __call__(self, inputs):
        return self.__conv_block(inputs)


    def __conv_block(self, inputs):
        max_pool = MaxPooling2D(pool_size=3, strides=1, padding='same')(inputs)
        dilation_conv = Conv2D(self.__filters, 3, padding='same', activation='relu'
                        , dilation_rate=self.__dilation_rate)(max_pool)

        conv = Conv2D(self.__filters, 1, padding='same', activation='relu')(dilation_conv)
        return conv
