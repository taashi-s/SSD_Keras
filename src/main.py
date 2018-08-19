import os
import numpy as np
from matplotlib import pyplot
import keras.callbacks as KC
import math

from ssd import SSD
from images_loader import load_images, save_images
from option_parser import get_option
from data_generator import DataGenerator
from history_checkpoint_callback import HistoryCheckpoint


CLASS_NUM = 4
INPUT_IMAGE_SHAPE = (300, 300, 3)
BATCH_SIZE = 45
EPOCHS = 1000
GPU_NUM = 1 # 4

INTERNAL_FILTER = 64

DIR_BASE = os.path.join('.', '..')
DIR_MODEL = os.path.join(DIR_BASE, 'model')
DIR_TRAIN_INPUTS = os.path.join(DIR_BASE, 'inputs')
DIR_TRAIN_TEACHERS = os.path.join(DIR_BASE, 'teachers')
DIR_VALID_INPUTS = os.path.join(DIR_BASE, 'valid_inputs')
DIR_VALID_TEACHERS = os.path.join(DIR_BASE, 'valid_teachers')
DIR_OUTPUTS = os.path.join(DIR_BASE, 'outputs')
DIR_TEST = os.path.join(DIR_BASE, 'predict_data')
DIR_PREDICTS = os.path.join(DIR_BASE, 'predict_data')

File_MODEL = 'segmentation_model.hdf5'


def train(gpu_num=None, with_generator=False, load_model=False, show_info=True):
    print('network creating ... ', end='', flush=True)
    network = SSD(INPUT_IMAGE_SHAPE, class_num=CLASS_NUM)
    print('... created')

    if show_info:
        network.plot_model_summary('../model_plot.png')
        network.show_model_summary()
    if isinstance(gpu_num, int):
        model = network.get_parallel_model(gpu_num, with_comple=True)
    else:
        model = network.get_model(with_comple=True)

    model_filename = os.path.join(DIR_MODEL, File_MODEL)
    callbacks = [ KC.TensorBoard()
                , HistoryCheckpoint(filepath='LearningCurve_{history}.png'
                                    , verbose=1
                                    , period=10
                                   )
                , KC.ModelCheckpoint(filepath=model_filename
                                     , verbose=1
                                     , save_weights_only=True
                                     , save_best_only=True
                                     , period=10
                                    )
                ]

    if load_model:
        print('loading weghts ... ', end='', flush=True)
        model.load_weights(model_filename)
        print('... loaded') 

    print('data generator creating ... ', end='', flush=True)
    train_generator = DataGenerator(DIR_TRAIN_INPUTS, DIR_TRAIN_TEACHERS, INPUT_IMAGE_SHAPE)
    valid_generator = DataGenerator(DIR_VALID_INPUTS, DIR_VALID_TEACHERS, INPUT_IMAGE_SHAPE)
    print('... created')

    if with_generator:
        train_data_num = train_generator.data_size()
        valid_data_num = valid_generator.data_size()
        history = model.fit_generator(train_generator.generator(batch_size=BATCH_SIZE)
                                      , steps_per_epoch=math.ceil(train_data_num / BATCH_SIZE)
                                      , epochs=EPOCHS
                                      , verbose=1
                                      , use_multiprocessing=True
                                      , callbacks=callbacks
                                      , validation_data=valid_generator.generator(batch_size=BATCH_SIZE)
                                      , validation_steps=math.ceil(valid_data_num / BATCH_SIZE)
                                     )
    else:
        print('data generateing ... ') #, end='', flush=True)
        train_inputs, train_teachers = train_generator.generate_data()
        valid_data = valid_generator.generate_data()
        print('... generated')
        history = model.fit(train_inputs, train_teachers, batch_size=BATCH_SIZE, epochs=EPOCHS
                            , validation_data=valid_data
                            , shuffle=True, verbose=1, callbacks=callbacks)
    print('model saveing ... ', end='', flush=True)
    model.save_weights(model_filename)
    print('... saved')
    print('learning_curve saveing ... ', end='', flush=True)
    save_learning_curve(history)
    print('... saved')


def save_learning_curve(history):
    """ save_learning_curve """
    x = range(EPOCHS)
    pyplot.plot(x, history.history['loss'], label="loss")
    pyplot.title("loss")
    pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    lc_name = 'LearningCurve'
    pyplot.savefig(lc_name + '.png')
    pyplot.close()


def predict(input_dir, gpu_num=None):
    (file_names, inputs) = load_images(input_dir, INPUT_IMAGE_SHAPE)
    network = SSD(INPUT_IMAGE_SHAPE, class_num=CLASS_NUM)
    if isinstance(gpu_num, int):
        model = network.get_parallel_model(gpu_num)
    else:
        model = network.get_model()
#    model.summary()
    print('loading weghts ...')
    model.load_weights(os.path.join(DIR_MODEL, File_MODEL))
    print('... loaded')
    print('predicting ...')
    preds = model.predict(inputs, BATCH_SIZE)
    print('... predicted')

    print('result saveing ...')
    image_data = __outputs_to_image_data(inputs, preds)
    save_images(DIR_OUTPUTS, image_data, file_names)
    print('... finish .')


def __outputs_to_image_data(images, preds):
    # TODO : Update
    image_data = []
    for i, img in enumerate(images):
        # Parse the outputs.
        det_label = preds[i][:, 0]
        det_conf = preds[i][:, 1]
        det_xmin = preds[i][:, 2]
        det_ymin = preds[i][:, 3]
        det_xmax = preds[i][:, 4]
        det_ymax = preds[i][:, 5]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        colors = plt.cm.hsv(np.linspace(0, 1, 4)).tolist()

        plt.imshow(img / 255.)
        currentAxis = plt.gca()

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            display_txt = '{:0.2f}, {}'.format(score, label)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
        
        plt.show()
    return image_data


if __name__ == '__main__':
    args = get_option(EPOCHS)
    EPOCHS = args.epoch

    if not(os.path.exists(DIR_MODEL)):
        os.mkdir(DIR_MODEL)
    if not(os.path.exists(DIR_OUTPUTS)):
        os.mkdir(DIR_OUTPUTS)

    train(gpu_num=GPU_NUM, with_generator=False, load_model=False)
    #train(gpu_num=GPU_NUM, with_generator=True, load_model=False)

    #predict(DIR_INPUTS, gpu_num=GPU_NUM)
    #predict(DIR_PREDICTS, gpu_num=GPU_NUM)
