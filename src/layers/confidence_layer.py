from keras.layers import Conv2D, Flatten, Dense


class ConfidenceLayer():
    def __init__(self, class_num, priors=6, use_dense=False):
        self.__class_num = class_num
        self.__priors = priors
        self.__use_dense = use_dense


    def __call__(self, inputs):
        if self.__use_dense:
            return self.__confidence_layer_dense(inputs)
        else:
            return self.__confidence_layer_conv(inputs)


    def __confidence_layer_dense(self, inputs):
        location = Dense(self.__priors * self.__class_num)(inputs)
        return location


    def __confidence_layer_conv(self, inputs):
        conv = Conv2D(self.__priors * self.__class_num, 3, padding='same')(inputs)
        location = Flatten()(conv)
        return location
