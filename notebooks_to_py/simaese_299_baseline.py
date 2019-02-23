#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

sys.path.append('../py_scripts')

# In[2]:


from model_utils import create_simaese_model
import keras
import os
from model_utils import get_mobilenet_model, get_callbacks
from utils import get_train_test_data_dict, DictImageDataGenerator, preprocess_func

# define train params
LABEL_CNT = 5004
ALL_DATA_JSON = '../data/train_data.json'
BATCH_SIZE = 12
print('load done')
IMAGE_SHAPE = (299, 299)
POS_RATIO = 0.5

# In[3]:


# get generator
train_d, val_d = get_train_test_data_dict(ALL_DATA_JSON, 0.2, use_new_whale=False, duplicate_low_cls=False)

# simple check
cnt = 0
for k in val_d:
    if len(val_d[k]) > 0:
        cnt += 1
print('val cls cnt', cnt)
cnt = 0
for k in train_d:
    if len(train_d[k]) > 0:
        cnt += 1
print('train cls cnt', cnt)

# In[4]:


train_ds = DictImageDataGenerator(rotation_range=10,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  horizontal_flip=True,
                                  preprocessing_function=preprocess_func)
train_gen = train_ds.flow_from_dict_for_simaese(train_d, target_size=IMAGE_SHAPE, batch_size=BATCH_SIZE,
                                                pos_ratio=POS_RATIO)

val_ds = DictImageDataGenerator(preprocessing_function=preprocess_func)
val_gen = val_ds.flow_from_dict_for_simaese(val_d, target_size=IMAGE_SHAPE, batch_size=BATCH_SIZE,
                                            pos_ratio=POS_RATIO)

val_steps = val_gen.samples // BATCH_SIZE
print(val_steps)
# test
for [x1, x2], y in train_gen:
    print(x1.shape, x2.shape, y.shape)
    break
for [x1, x2], y in val_gen:
    print(x1.shape, x2.shape, y.shape)
    print(y)
    break

# In[ ]:


all_model, feat_model, compare_model = create_simaese_model(img_shape=IMAGE_SHAPE, mid_feat_dim=256, mob_alpha=0.75)

print(all_model.input_shape, all_model.output_shape)
all_model.summary()

# In[ ]:


# get generator and train
cb_list = get_callbacks('../data/checkpoints/mob_299_sim.h5', all_model)
adam_opt = keras.optimizers.Adam(lr=0.0001)
all_model.compile(optimizer=adam_opt,
                  loss='binary_crossentropy',
                  metrics=['acc'])
print('compile done')

# In[ ]:


# train
all_model.fit_generator(
    train_gen,
    steps_per_epoch=1000,
    epochs=100,
    verbose=1,
    callbacks=cb_list,
    validation_data=val_gen,
    validation_steps=val_steps
)
