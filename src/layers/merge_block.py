import keras.backend as KB
from keras.layers import Concatenate, Reshape, Activation


class MergeBlock():
    def __init__(self):
        pass


    def __call__(self, inputs):
        return self.__merge_block(inputs)


    def __merge_block(self, inputs):
        loc_layers = []
        conf_layers = []
        pbox_layers = []

        for loc, conf, pbox in inputs:
            loc_layers.append(loc)
            conf_layers.append(conf)
            pbox_layers.append(pbox)

        merge_loc = Concatenate(axis=1)(loc_layers)
        box_num = KB.shape(merge_loc)[-1] // 4
        locs = Reshape((box_num, 4))(merge_loc)

        merge_conf = Concatenate(axis=1)(conf_layers)
        reshape_conf = Reshape((box_num, 4))(merge_conf)
        confs = Activation('softmax')(reshape_conf)

        pboxs = Concatenate(axis=1)(pbox_layers)

        merge_layer = Concatenate(axis=2)([locs, confs, pboxs])
        return merge_layer
