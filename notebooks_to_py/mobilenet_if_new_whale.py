#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

sys.path.append('../py_scripts')

# In[2]:


import keras
import os
from model_utils import get_mobilenet_model, get_callbacks
from utils import get_train_test_data_dict, DictImageDataGenerator, preprocess_func
from utils import get_if_new_whale_dict, split_train_test_dict

# define train params
IMG_SIZE = 224
LABEL_CNT = 2
ALL_DATA_JSON = '../data/train_data.json'
BATCH_SIZE = 16
ALL_DATA_DICT = json.loads(open(ALL_DATA_JSON).read())
print('load done')

if_whale_d = get_if_new_whale_dict(ALL_DATA_DICT)
train_d, val_d = split_train_test_dict(if_whale_d)
print(len(train_d['new_whale']))
print(len(val_d['new_whale']))

# In[3]:


train_ds = DictImageDataGenerator(rotation_range=20,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  zoom_range=0.2,
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

# In[4]:


mob_model = get_mobilenet_model(IMG_SIZE, LABEL_CNT, dense_dim=64)
mob_model.summary()

# In[5]:


cb_list = get_callbacks('../data/checkpoints/detect_if_new_whale.h5', mob_model)
adam_opt = keras.optimizers.Adam(lr=0.001)
mob_model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['acc'])
print('compile done')

# In[6]:


# train
mob_model.fit_generator(
    train_gen,
    steps_per_epoch=2000,
    epochs=100,
    verbose=1,
    callbacks=cb_list,
    validation_data=val_gen,
    validation_steps=val_steps
)
