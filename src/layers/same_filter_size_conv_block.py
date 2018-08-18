from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, BatchNormalization


class SameFilterSizeConvBlock():
    def __init__(self, filters, conv_num=2, with_max_pool=True):
        self.__filters = filters
        self.__conv_num = conv_num
        self.__with_max_pool = with_max_pool


    def __call__(self, inputs):
        return self.__conv_block(inputs)


    def __conv_block(self, inputs):
        layer = inputs
        if self.__with_max_pool:
           layer = MaxPooling2D(padding='same')(layer)

        for _ in range(self.__conv_num):
            layer = Conv2D(self.__filters, 3, padding='same', activation='relu')(layer)

        return layer

