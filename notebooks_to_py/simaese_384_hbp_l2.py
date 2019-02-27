#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

sys.path.append('../py_scripts')

# In[2]:


from model_utils import create_simaese_model
import keras
import os
import json
from model_utils import get_mobilenet_model, get_callbacks
from utils import get_train_test_data_dict, DictImageDataGenerator, preprocess_func

# define train params
LABEL_CNT = 5004
ALL_DATA_JSON = '../data/train_data.json'
BATCH_SIZE = 12
print('load done')
IMAGE_SHAPE = (384, 384)
POS_RATIO = 0.5
VAL_STEP = 50  # use small val step

# In[3]:


# get generator
train_d, val_d = get_train_test_data_dict(ALL_DATA_JSON, 0.3, use_new_whale=False, duplicate_low_cls=True)
# use all data in train,
train_d = json.loads(open(ALL_DATA_JSON).read())
train_d.pop('new_whale')
# drop 1 image class
all_labels = list(train_d.keys())
for k in all_labels:
    if len(train_d[k]) == 1:
        train_d.pop(k)
    if len(val_d[k]) == 1:
        val_d.pop(k)

# simple check
cnt = 0
for k in val_d:
    if len(val_d[k]) > 0:
        cnt += 1
print('val cls cnt', cnt)

# In[4]:
train_ds = DictImageDataGenerator(rotation_range=10,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  horizontal_flip=True,
                                  preprocessing_function=preprocess_func)
train_gen = train_ds.flow_from_dict_for_simaese(train_d, target_size=IMAGE_SHAPE, batch_size=BATCH_SIZE,
                                                pos_ratio=POS_RATIO)
# add some aug in val
val_ds = DictImageDataGenerator(width_shift_range=0.05,
                                height_shift_range=0.05,
                                preprocessing_function=preprocess_func)
val_gen = val_ds.flow_from_dict_for_simaese(val_d, target_size=IMAGE_SHAPE, batch_size=BATCH_SIZE,
                                            pos_ratio=POS_RATIO)

# test
for [x1, x2], y in train_gen:
    print(x1.shape, x2.shape, y.shape)
    break
for [x1, x2], y in val_gen:
    print(x1.shape, x2.shape, y.shape)
    print(y)
    break

# In[ ]:


all_model, feat_model, compare_model = create_simaese_model(img_shape=IMAGE_SHAPE, mid_feat_dim=512, mob_alpha=0.75,
                                                            head_model_name='mobilenet_hbp', l2_flag=True)

print(all_model.input_shape, all_model.output_shape)
all_model.summary()

# In[ ]:


# get generator and train
cb_list = get_callbacks('../data/checkpoints/mob_384_sim_hbp.h5', all_model)
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
    validation_steps=VAL_STEP
)

# Epoch 00004: ReduceLROnPlateau reducing learning rate to 4.999999987376214e-07.
#
# Epoch 00004: val_loss improved from 0.11668 to 0.11243, saving model to ../data/checkpoints/mob_299_sim_hbp.h5
#
# Epoch 00004: loss did not improve from 0.15524
# Epoch 5/100
# 1000/1000 [==============================] - 2807s 3s/step - loss: 0.1591 - acc: 0.9376 - val_loss: 0.1202 - val_acc: 0.9528
