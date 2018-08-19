from keras.layers import Conv2D, ZeroPadding2D
from keras.layers import Activation, BatchNormalization


class UpFilterSizeConvBlock():
    def __init__(self, first_filters, with_zero_pad=False):
        self.__first_filters = first_filters
        self.__with_zero_pad = with_zero_pad


    def __call__(self, inputs):
        return self.__conv_block(inputs)


    def __conv_block(self, inputs):
        filters = self.__first_filters
        conv_1 = Conv2D(filters, 1, padding='same', activation='relu')(inputs)

        padding = 'same'
        if self.__with_zero_pad:
            conv_1 = ZeroPadding2D()(conv_1)
            padding = 'valid'

        filters *= 2
        conv_2 = Conv2D(filters, 3, strides=2, padding=padding, activation='relu')(conv_1)

        return conv_2
