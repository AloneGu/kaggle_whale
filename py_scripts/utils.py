#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: utils.py
@time: 2/13/19 20:56
"""
import os
import numpy as np
from keras import backend
from keras.preprocessing.image import ImageDataGenerator, Iterator, load_img, img_to_array, array_to_img
from scipy.misc import imresize


class DictImageDataGenerator(ImageDataGenerator):
    def flow_from_dict(self, cls_fp_dict,
                       target_size=(256, 256), color_mode='rgb',
                       classes=None, class_mode='categorical',
                       batch_size=32, shuffle=True, seed=None,
                       interpolation='nearest'):
        # cls_fp_dict {'label1':[fp1,fp2], 'label2':[fp3,fp4]}
        return FlDictIterator(
            cls_fp_dict, self, target_size=target_size, color_mode=color_mode, class_mode=class_mode,
            batch_size=batch_size, interpolation=interpolation, shuffle=shuffle, seed=seed
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
                 interpolation='nearest'):
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
            img = load_img(os.path.join(self.directory, fname),
                           color_mode=self.color_mode
                           )  # resize later
            x = img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(x, params)
            x = imresize(x, size=self.target_size, interp=self.interpolation) # resize after random transforms
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
