from . import NormalizeLayer, LocationLayer, ConfidenceLayer, PriorBoxLayer

class RegionBlock():
    def __init__(self, class_num, priors=6, with_normalize=False, use_dense=False):
        self.__class_num = class_num
        self.__priors = priors
        self.__with_normalize = with_normalize
        self.__use_dense = use_dense


    def __call__(self, inputs):
        return self.__region_block(inputs)


    def __region_block(self, inputs):
        layer = inputs
        if self.__with_normalize:
            layer = NormalizeLayer()(layer)

        loc = LocationLayer(priors=self.__priors
                            , use_dense=self.__use_dense)(layer)
        conf = ConfidenceLayer(self.__class_num, priors=self.__priors
                               , use_dense=self.__use_dense)(layer)
        priorbox = PriorBoxLayer()(layer)

        return loc, conf, priorbox
