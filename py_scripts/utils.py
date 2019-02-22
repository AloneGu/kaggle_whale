#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: utils.py
@time: 2/13/19 20:56
"""
import os
import random
import numpy as np
import json
import pandas as pd
from keras import backend
from keras.preprocessing.image import ImageDataGenerator, Iterator, load_img, img_to_array, array_to_img
from scipy.misc import imresize

CURR_DIR = os.path.abspath(os.path.join(__file__, '..'))
BBOX_PATH = os.path.join(CURR_DIR, '../data/bounding_boxes.csv')
BBOX_DF = pd.read_csv(BBOX_PATH).set_index('Image')


def get_bbox(fp):
    bbox = BBOX_DF.loc[fp]
    x0, y0, x1, y1 = bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1']
    return x0, y0, x1, y1


def preprocess_func(x):
    return x / 127.5 - 1.0


class DictImageDataGenerator(ImageDataGenerator):
    def flow_from_dict(self, cls_fp_dict,
                       target_size=(256, 256), color_mode='rgb',
                       classes=None, class_mode='categorical',
                       batch_size=32, shuffle=True, seed=None,
                       interpolation='nearest', use_bbox=True):
        # cls_fp_dict {'label1':[fp1,fp2], 'label2':[fp3,fp4]}
        return FlDictIterator(
            cls_fp_dict, self, target_size=target_size, color_mode=color_mode, class_mode=class_mode,
            batch_size=batch_size, interpolation=interpolation, shuffle=shuffle, seed=seed, use_bbox=True
        )

    def flow_from_dict_for_simaese(self, cls_fp_dict,
                                   target_size=(256, 256), color_mode='rgb',
                                   classes=None, class_mode='categorical',
                                   batch_size=32, shuffle=True, seed=None,
                                   interpolation='nearest', use_bbox=True, pos_ratio=0.5):
        return FlSimaeseDictIterator(
            cls_fp_dict, self, target_size=target_size, color_mode=color_mode, class_mode=class_mode,
            batch_size=batch_size, interpolation=interpolation,
            shuffle=shuffle, seed=seed, use_bbox=use_bbox, pos_ratio=pos_ratio
        )


class FlDictIterator(Iterator):
    def __init__(self, cls_fp_dict, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest', use_bbox=True):
        if data_format is None:
            data_format = backend.image_data_format()
        self.cls_fp_dict = cls_fp_dict
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'rgba', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb", "rgba", or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgba':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (4,)
            else:
                self.image_shape = (4,) + self.target_size
        elif self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation

        # First, count the number of samples and classes.
        if not classes:  # classes names
            classes = sorted(list(self.cls_fp_dict.keys()))
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))
        self.samples = 0
        for k in self.cls_fp_dict:
            self.samples += len(self.cls_fp_dict[k])

        print('Found %d images belonging to %d classes.' %
              (self.samples, self.num_classes))

        # Second, build an index of the images
        # in the different class subfolders.
        self.filenames = []
        self.classes = []
        for k in self.cls_fp_dict:
            fp_list = self.cls_fp_dict[k]
            cls_idx = self.class_indices[k]
            for i, tmp_fp in enumerate(fp_list):
                self.filenames.append(tmp_fp)
                self.classes.append(cls_idx)
        self.classes = np.array(self.classes)

        # flag for bbox
        self.use_bbox = use_bbox
        if self.use_bbox:
            print('Using bbox for image')

        super(FlDictIterator, self).__init__(self.samples,
                                             batch_size,
                                             shuffle,
                                             seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=backend.floatx())
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(fname,
                           color_mode=self.color_mode
                           )  # resize later
            if self.use_bbox is True:
                base_fn = os.path.basename(fname)
                x0, y0, x1, y1 = get_bbox(base_fn)
                if not (x0 >= x1 or y0 >= y1):
                    tmp_box = (x0, y0, x1, y1)
                    img.crop(tmp_box)
            x = img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(x, params)
            x = imresize(x, size=self.target_size, interp=self.interpolation)  # resize after random transforms
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(backend.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(
                (len(batch_x), self.num_classes),
                dtype=backend.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


class FlSimaeseDictIterator(Iterator):
    def __init__(self, cls_fp_dict, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 use_bbox=True,
                 pos_ratio=0.5  # positive pairs in each batch
                 ):
        if data_format is None:
            data_format = backend.image_data_format()
        self.cls_fp_dict = cls_fp_dict
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'rgba', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb", "rgba", or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgba':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (4,)
            else:
                self.image_shape = (4,) + self.target_size
        elif self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        self.pos_ratio = pos_ratio

        # First, count the number of samples and classes.
        if not classes:  # classes names
            classes = sorted(list(self.cls_fp_dict.keys()))
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))
        self.samples = 0
        for k in self.cls_fp_dict:
            self.samples += len(self.cls_fp_dict[k])

        print('Found %d images belonging to %d classes.' %
              (self.samples, self.num_classes))

        # Second, build an index of the images
        # in the different class subfolders.
        self.all_labels = list(self.cls_fp_dict.keys())
        self.filename_to_cls_label = {}
        self.filenames = []
        self.classes = []
        for k in self.cls_fp_dict:  # k is string label like new_whale
            fp_list = self.cls_fp_dict[k]
            cls_idx = self.class_indices[k]  # cls_idx is integer like 123
            for i, tmp_fp in enumerate(fp_list):
                self.filenames.append(tmp_fp)
                self.classes.append(cls_idx)
                self.filename_to_cls_label[tmp_fp] = k  # save label to find same or diff images
        self.classes = np.array(self.classes)

        # flag for bbox
        self.use_bbox = use_bbox
        if self.use_bbox:
            print('Using bbox for image')

        super(FlSimaeseDictIterator, self).__init__(self.samples,
                                                    batch_size,
                                                    shuffle,
                                                    seed)

    def _get_batches_of_transformed_samples(self, index_array):
        # init x
        batch_size = len(index_array)
        batch_x1 = np.zeros(
            (batch_size,) + self.image_shape,
            dtype=backend.floatx())
        batch_x2 = np.zeros(
            (batch_size,) + self.image_shape,
            dtype=backend.floatx())

        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(fname,
                           color_mode=self.color_mode
                           )  # resize later
            if self.use_bbox is True:
                base_fn = os.path.basename(fname)
                x0, y0, x1, y1 = get_bbox(base_fn)
                if not (x0 >= x1 or y0 >= y1):
                    tmp_box = (x0, y0, x1, y1)
                    img.crop(tmp_box)
            x = img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(x, params)
            x = imresize(x, size=self.target_size, interp=self.interpolation)  # resize after random transforms
            x = self.image_data_generator.standardize(x)
            batch_x1[i] = x

        # build next batch
        pos_cnt = int(batch_size * self.pos_ratio)
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            curr_label = self.filename_to_cls_label[fname]
            if j <= pos_cnt:  # find same cls
                this_label_fps = self.cls_fp_dict[curr_label]
                fname = random.choice(this_label_fps)
            else:  # random find another cls
                other_labels = [l for l in self.all_labels if l != curr_label]
                rnd_label = random.choice(other_labels)
                other_label_fps = self.cls_fp_dict[rnd_label]
                fname = random.choice(other_label_fps)
            img = load_img(fname,
                           color_mode=self.color_mode
                           )  # resize later
            if self.use_bbox is True:
                base_fn = os.path.basename(fname)
                x0, y0, x1, y1 = get_bbox(base_fn)
                if not (x0 >= x1 or y0 >= y1):
                    tmp_box = (x0, y0, x1, y1)
                    img.crop(tmp_box)
            x = img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(x, params)
            x = imresize(x, size=self.target_size, interp=self.interpolation)  # resize after random transforms
            x = self.image_data_generator.standardize(x)
            batch_x2[i] = x

        # build batch of labels
        batch_y = np.zeros(shape=(batch_size, 1), dtype=backend.floatx())
        batch_y[:pos_cnt, :] = 1.0

        return [batch_x1, batch_x2], batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


def split_train_test_dict(all_data_d, test_rate=0.2, duplicate_low_cls=True):
    train_d = {}
    val_d = {}
    for k in all_data_d:
        fp_list = all_data_d[k]
        fp_cnt = len(fp_list)
        split_idx = int(fp_cnt * test_rate)
        if split_idx == 0:  # low image count
            if duplicate_low_cls:
                train_d[k] = fp_list
                val_d[k] = fp_list
            else:
                train_d[k] = fp_list
                val_d[k] = []
        else:
            train_d[k] = fp_list[split_idx:]
            val_d[k] = fp_list[:split_idx]
    return train_d, val_d


def get_train_test_data_dict(json_path, test_rate=0.2, use_new_whale=True, duplicate_low_cls=True):
    all_data_d = json.loads(open(json_path).read())
    if use_new_whale is False:
        all_data_d.pop('new_whale')
    return split_train_test_dict(all_data_d, test_rate, duplicate_low_cls)


def get_if_new_whale_dict(all_data_d):
    new_d = {}
    new_d['not_new_whale'] = []
    new_d['new_whale'] = all_data_d['new_whale']
    for k in all_data_d:
        if k != 'new_whale':
            new_d['not_new_whale'] += all_data_d[k]
    return new_d
