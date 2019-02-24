#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: fgvc_models.py
@time: 2/24/19 21:50
"""

# BCNN http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf
# HBP http://openaccess.thecvf.com/content_ECCV_2018/papers/Chaojian_Yu_Hierarchical_Bilinear_Pooling_ECCV_2018_paper.pdf

import keras
from keras import backend as K
from keras.applications.mobilenet import MobileNet
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2


def out_product(x):
    return K.batch_dot(x[0], x[1], axes=[1, 1]) / x[0].get_shape().as_list()[1]


def signed_sqrt(x):
    return K.sign(x) * K.sqrt(K.abs(x) + K.epsilon())


def L2_norm(x, axis=-1):
    return K.l2_normalize(x, axis=axis)


def inner_sum(x):
    return K.sum(x, axis=1)


def HBP_cross_feat(x1, x2, feat_dim, project_num):
    feat = keras.layers.Multiply()([x1, x2])
    feat = keras.layers.Reshape([feat_dim * feat_dim, project_num])(feat)
    feat = keras.layers.Lambda(inner_sum)(feat)
    feat = keras.layers.Lambda(signed_sqrt)(feat)
    feat = keras.layers.Lambda(L2_norm)(feat)
    return feat


def BCNN_mobilenet(image_shape, feat_conv_num=256):
    pass


def HBP_mobilenet(image_shape=(448, 448), feat_dim=28, project_num=8192, alpha=1.0):
    # feat_dim should set based on image shape
    tensor_inp = keras.layers.Input(shape=[image_shape[0], image_shape[1], 3])  # use 3 channels
    base_model = MobileNet(include_top=False, weights=None, input_tensor=tensor_inp, alpha=alpha)

    # hbp
    feat_1 = base_model.layers[73]
    feat_2 = base_model.layers[67]
    feat_3 = base_model.layers[61]
    bilinear_1 = keras.layers.Conv2D(project_num, 1, padding='same')(feat_1.output)
    bilinear_2 = keras.layers.Conv2D(project_num, 1, padding='same')(feat_2.output)
    bilinear_3 = keras.layers.Conv2D(project_num, 1, padding='same')(feat_3.output)
    cross_1 = HBP_cross_feat(bilinear_1, bilinear_2, feat_dim, project_num)
    cross_2 = HBP_cross_feat(bilinear_1, bilinear_3, feat_dim, project_num)
    cross_3 = HBP_cross_feat(bilinear_2, bilinear_3, feat_dim, project_num)
    final_out = keras.layers.concatenate([cross_1, cross_2, cross_3])
    return keras.models.Model(tensor_inp, final_out)


if __name__ == '__main__':
    test_m = HBP_mobilenet(image_shape=(299, 299), feat_dim=18, project_num=2048, alpha=0.75)
    test_m.summary()
    # test_m = HBP_mobilenet(project_num=1024)
    # test_m.summary()
