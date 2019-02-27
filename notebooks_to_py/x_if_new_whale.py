#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys

sys.path.append('../py_scripts')

import keras
import json
import os
from model_utils import get_mobilenet_model, get_callbacks, get_xception_model
from utils import get_train_test_data_dict, DictImageDataGenerator, preprocess_func
from utils import get_if_new_whale_dict, split_train_test_dict

# define train params
IMG_SIZE = 299
LABEL_CNT = 2
ALL_DATA_JSON = '../data/train_data.json'
BATCH_SIZE = 12
ALL_DATA_DICT = json.loads(open(ALL_DATA_JSON).read())
print('load done')

if_whale_d = get_if_new_whale_dict(ALL_DATA_DICT)
train_d, val_d = split_train_test_dict(if_whale_d, test_rate=0.075)
# add val_d to train_d
train_d = if_whale_d
print(len(train_d['new_whale']), len(train_d['not_new_whale']))
print(len(val_d['new_whale']), len(val_d['not_new_whale']))

train_ds = DictImageDataGenerator(rotation_range=20,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  zoom_range=0.1,
                                  horizontal_flip=True,
                                  preprocessing_function=preprocess_func)
train_gen = train_ds.flow_from_dict(train_d, target_size=(
    IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE)
val_ds = DictImageDataGenerator(preprocessing_function=preprocess_func)
val_gen = val_ds.flow_from_dict(val_d, target_size=(
    IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE)
val_steps = val_gen.samples // BATCH_SIZE
print(val_steps)
# test
for x, y in train_gen:
    print(x.shape, y.shape)
    break
print(train_gen.class_indices)

mob_model = get_xception_model(IMG_SIZE, LABEL_CNT, dense_dim=None)
# mob_model.summary()

cb_list = get_callbacks('../data/checkpoints/detect_if_new_whale_bbox_xception.h5', mob_model)
adam_opt = keras.optimizers.Adam(lr=0.00001)
mob_model.compile(optimizer=adam_opt,
                  loss='categorical_crossentropy', metrics=['acc'])
print('compile done')

# train
mob_model.fit_generator(
    train_gen,
    steps_per_epoch=1000,
    epochs=100,
    verbose=1,
    callbacks=cb_list,
    validation_data=val_gen,
    validation_steps=val_steps
)
