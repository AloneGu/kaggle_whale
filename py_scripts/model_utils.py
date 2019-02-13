#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: model_utils.py
@time: 2/13/19 20:57
"""

import keras
from keras.layers import Input
from keras.applications import MobileNet, ResNet50, InceptionResNetV2, Xception


def get_mobilenet_model(img_size=224, label_cnt=5005, dense_dim=1024):
    """

    :param img_size: input img size
    :return: model structure
    """
    input_tensor = Input(shape=(img_size,img_size,3))
    base_model = MobileNet(input_tensor=input_tensor, include_top=False, weights='imagenet')
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.Dense(dense_dim, activation='relu')(x)
    x = keras.layers.Dense(label_cnt)(x)
    model = keras.Model(input_tensor,x)
    return model