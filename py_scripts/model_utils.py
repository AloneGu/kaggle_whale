#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: model_utils.py
@time: 2/13/19 20:57
"""
import os
import keras
from keras.layers import Input
from keras.applications import MobileNet, ResNet50, InceptionResNetV2, Xception


def get_mobilenet_model(img_size=224, label_cnt=5005, dense_dim=1024):
    """

    :param img_size: input img size
    :return: model structure
    """
    if isinstance(img_size, list):  # 224,100
        h, w = img_size
        input_tensor = Input(shape=(h, w, 3))
    else:
        input_tensor = Input(shape=(img_size, img_size, 3))
    base_model = MobileNet(input_tensor=input_tensor, include_top=False, weights='imagenet')
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    if dense_dim is not None:
        x = keras.layers.Dense(dense_dim, activation='relu')(x)
    x = keras.layers.Dense(label_cnt, activation='softmax')(x)
    model = keras.Model(input_tensor, x)
    return model


def get_callbacks(model_save_path, model=None):
    if os.path.exists(model_save_path):
        if model is not None:
            model.load_weights(model_save_path, skip_mismatch=True, by_name=True)
            print('load pre weights')
    model_dir = os.path.dirname(model_save_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    m_reduce = keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=3, verbose=1, min_lr=0.00000001)
    m_check = keras.callbacks.ModelCheckpoint(
        model_save_path, monitor='val_loss', verbose=1, save_best_only=True)
    return [m_reduce, m_check]
