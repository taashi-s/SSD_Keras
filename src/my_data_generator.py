import os
import glob
import cv2
import random
import numpy as np
import time

import images_loader as iml

class DataGenerator(object):
    def __init__(self, input_dir, teacher_dir, bbox_util, image_shape, with_norm=True):
        self.__input_dir = input_dir
        self.__teacher_dir = teacher_dir
        self.__bbox_util = bbox_util
        self.__image_shape = image_shape
        self.__with_norm = with_norm
        self.__data_names = self.__get_data_names(self.__input_dir, self.__teacher_dir)


    def __get_data_names(self, input_dir, teacher_dir):
        files = glob.glob(os.path.join(input_dir, '*.png'))
        files += glob.glob(os.path.join(input_dir, '*.jpeg'))
        files += glob.glob(os.path.join(input_dir, '*.jpg'))
        files.sort()

        data_names = []
        for file in files:
            # TODO : Support other extension
            name = os.path.basename(file)
            basename, _ = os.path.splitext(name)
            teacher_path = os.path.join(teacher_dir, basename + '.npy')
            if not os.path.exists(teacher_path):
                print('teacher file is not exists : ', teacher_path)
                continue
            data_names.append(name)
        return data_names


    def data_size(self):
        return len(self.__data_names)


    def generate_data(self, target_data_list=None, batch_size=None):
        data_list = self.__data_names
        if target_data_list is not None:
            data_list = target_data_list

        data_num = len(data_list)
        start = time.time()
        prev_time = start

        input_list = []
        teacher_list = []
        random.shuffle(data_list)
        for k, name in enumerate(data_list):
            input_img, teacher_vecs = self.load_data(name)
            if input_img is None or teacher_vecs is None:
                continue

            input_list.append(input_img)
            teacher_list.append(teacher_vecs)
            if k % 200 == 0 or k == data_num - 1:
                now_time = time.time()
                print('## generate : ', '%05d' % k, '/', '%05d' % data_num, ' %5.3f(%5.3f)' % (now_time - prev_time, now_time - start))
                prev_time = now_time

        inputs = np.array(input_list)
        teachers = np.array(teacher_list)

        if batch_size is not None:
            for k in range(batch_size):
                remain = len(inputs) % batch_size
                if remain == 0:
                    break
                diff = batch_size - remain
                inputs = np.concatenate([inputs, inputs[:diff]], axis=0)
                teachers = np.concatenate([teachers, teachers[:diff]], axis=0)
        print('$$$ np.shape(inputs) : ', np.shape(inputs))
        return inputs, teachers


    def generator(self, batch_size=None, target_data_list=None):
        data_list = self.__data_names
        if target_data_list is not None:
            data_list = target_data_list

        if batch_size is None:
            batch_size = self.data_size()

        input_list = []
        teacher_list = []

        while True:
            random.shuffle(data_list)
            for name in data_list:
                input_img, teacher_vecs = self.load_data(name)
                if input_img is None or teacher_vecs is None:
                    continue

                input_list.append(input_img)
                teacher_list.append(teacher_vecs)

                if len(input_list) >= batch_size:
                    inputs = [np.array(input_list)]
                    teachers = [np.array(teacher_list)]
                    input_list = []
                    teacher_list = []
                    yield inputs, teachers


    def load_data(self, name):
        basename, _ = os.path.splitext(name)
        input_path = os.path.join(self.__input_dir, name)
        teacher_path = os.path.join(self.__teacher_dir, basename + '.npy')

        input_img = iml.load_image(input_path, self.__image_shape, with_normalize=self.__with_norm)
        teacher_vecs = np.load(teacher_path)

        if teacher_vecs is None:
            return input_img, None

        cls_one_hots = []
        bounding_boxes = []
        #cls_one_hots, bounding_boxes = zip(*teacher_vecs)
        for teacher_vec in teacher_vecs:
            cls_one_hot, bounding_box = teacher_vec
            cls_one_hot = cls_one_hot.tolist()
            bounding_box = bounding_box.tolist()
            cls_one_hots.append(cls_one_hot[1:])
            bounding_boxes.append(bounding_box)
        teacher_vecs_restack = np.hstack((np.asarray(bounding_boxes), np.asarray(cls_one_hots)))
        teacher_vecs_assign = self.__bbox_util.assign_boxes(teacher_vecs_restack)
        return input_img, teacher_vecs_assign
