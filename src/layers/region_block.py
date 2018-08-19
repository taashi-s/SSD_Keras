from . import NormalizeLayer, LocationLayer, ConfidenceLayer, PriorBoxLayer

class RegionBlock():
    def __init__(self, class_num, img_shape, min_size, max_size=None
                 , aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2]
                 , priors=6, normalize_scale=None, use_dense=False):
        self.__class_num = class_num
        self.__img_shape = img_shape
        self.__min_size = min_size
        self.__max_size = max_size
        self.__aspect_ratios = aspect_ratios
        self.__variances = variances
        self.__priors = priors
        self.__normalize_scale = normalize_scale
        self.__use_dense = use_dense


    def __call__(self, inputs):
        return self.__region_block(inputs)


    def __region_block(self, inputs):
        layer = inputs
        if isinstance(self.__normalize_scale, int):
            layer = NormalizeLayer(self.__normalize_scale)(layer)

        loc = LocationLayer(priors=self.__priors
                            , use_dense=self.__use_dense)(layer)
        conf = ConfidenceLayer(self.__class_num, priors=self.__priors
                               , use_dense=self.__use_dense)(layer)
        priorbox = PriorBoxLayer(self.__img_shape, self.__min_size
                                 , max_size=self.__max_size
                                 , aspect_ratios=self.__aspect_ratios
                                 , variances=self.__variances
                                )(layer)

        return loc, conf, priorbox
