from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.utils import multi_gpu_model, plot_model
import keras.backend as KB

from layers import SameFilterSizeConvBlock, UpFilterSizeConvBlock, DilationConvBlock
from layers import RegionBlock, MergeBlock, MergeBlock


class SSD(object):
    def __init__(self, input_shape, class_num=1):
        self.__input_shape = input_shape
        self.__class_num = class_num

        inputs = Input(self.__input_shape)
        conv1 = SameFilterSizeConvBlock(64, with_max_pool=False)(inputs)
        conv2 = SameFilterSizeConvBlock(128)(conv1)
        conv3 = SameFilterSizeConvBlock(256, conv_num=3)(conv2)
        conv4 = SameFilterSizeConvBlock(512, conv_num=3)(conv3)
        conv5 = SameFilterSizeConvBlock(512, conv_num=3)(conv4)

        conv6 = DilationConvBlock(1024, dilation_rate=(6, 6))(conv5)

        conv7 = UpFilterSizeConvBlock(256)(conv6)
        conv8 = UpFilterSizeConvBlock(128, with_zero_pad=True)(conv7)
        conv9 = UpFilterSizeConvBlock(128)(conv8)

        pool = GlobalAveragePooling2D()(conv9)

        region_1 = RegionBlock(class_num, input_shape, 30
                               , aspect_ratios=[2], priors=3, normalize_scale=20)(conv4)
        region_2 = RegionBlock(class_num, input_shape, 60, max_size=114)(conv6)
        region_3 = RegionBlock(class_num, input_shape, 114, max_size=168)(conv7)
        region_4 = RegionBlock(class_num, input_shape, 168, max_size=222)(conv8)
        region_5 = RegionBlock(class_num, input_shape, 222, max_size=276)(conv9)
        region_6 = RegionBlock(class_num, input_shape, 276, max_size=330
                               , use_dense=True)(pool)

        outputs = MergeBlock()([region_1, region_2, region_3, region_4, region_5, region_6])

        self.__model = Model(inputs=[inputs], outputs=[outputs])


    def comple_model(self):
        # TODO : loss function
        self.__model.compile(optimizer=Adam(lr=0.01), loss=[])


    def get_model(self, with_comple=False):
        if with_comple:
            self.comple_model()
        return self.__model


    def get_parallel_model(self, gpu_num, with_comple=False):
        self.__model = multi_gpu_model(self.__model, gpus=gpu_num)
        return self.get_model(with_comple)


    def show_model_summary(self):
        self.__model.summary()


    def plot_model_summary(self, file_name):
        plot_model(self.__model, to_file=file_name)
