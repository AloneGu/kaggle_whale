#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: model_utils.py
@time: 2/13/19 20:57
"""
import os
import keras
from keras import backend as K
from keras.layers import Lambda, Concatenate, Dense, Flatten
from keras.layers import Input, Conv2D, Reshape
from keras.applications import MobileNet, ResNet50, InceptionResNetV2, Xception
from fgvc_models import HBP_mobilenet


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
    m_check2 = keras.callbacks.ModelCheckpoint(
        model_save_path.replace('.h5', '_best_train_loss.h5'), monitor='loss', verbose=1, save_best_only=True)
    return [m_reduce, m_check, m_check2]


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


def get_xception_model(img_size=224, label_cnt=5005, dense_dim=1024):
    """

    :param img_size: input img size
    :return: model structure
    """
    if isinstance(img_size, list):  # 224,100
        h, w = img_size
        input_tensor = Input(shape=(h, w, 3))
    else:
        input_tensor = Input(shape=(img_size, img_size, 3))
    base_model = Xception(input_tensor=input_tensor, include_top=False, weights=None)
    x = keras.layers.GlobalMaxPooling2D()(base_model.output)  # diff use max pool
    if dense_dim is not None:
        x = keras.layers.Dense(dense_dim, activation='relu')(x)
    x = keras.layers.Dense(label_cnt, activation='softmax')(x)
    model = keras.Model(input_tensor, x)
    return model


def create_simaese_model(img_shape, mid_feat_dim=512, mid_compare_dim=128, head_model_name='mobilenet', mob_alpha=1.0):
    """

    :param img_shape: h, w
    :param mid_feat_dim: middle feat shape
    :param head_model_name: choose mobile net or others
    :return: all_model, feat_model, compare_model
    """

    # feat model
    img_input = Input(shape=(img_shape[0], img_shape[1], 3), name='img_feat_input')
    if head_model_name == 'mobilenet':
        # imagenet weights does not help
        feat_model = MobileNet(input_tensor=img_input, include_top=False, weights=None,
                               pooling='avg', alpha=mob_alpha)
        mid_feat = keras.layers.Dense(mid_feat_dim, name='img_feat_output', activation='sigmoid')(feat_model.output)
    elif head_model_name == 'mobilenet_hbp':
        feat_model = HBP_mobilenet(img_shape, feat_dim=18, project_num=2048, alpha=mob_alpha)
        img_input = feat_model.input  # updated input tensor
        mid_feat = keras.layers.Dense(mid_feat_dim, name='img_feat_output', activation='sigmoid')(feat_model.output)
    else:
        raise ValueError('head model name')

    feat_model = keras.Model(img_input, mid_feat, name='top_feat')

    # compare model
    xa_inp = Input(shape=feat_model.output_shape[1:], name='cmp_1')
    xb_inp = Input(shape=feat_model.output_shape[1:], name='cmp_2')
    x1 = Lambda(lambda x: x[0] * x[1])([xa_inp, xb_inp])
    x2 = Lambda(lambda x: x[0] + x[1])([xa_inp, xb_inp])
    x3 = Lambda(lambda x: K.abs(x[0] - x[1]))([xa_inp, xb_inp])
    x4 = Lambda(lambda x: K.square(x))(x3)
    x = Concatenate()([x1, x2, x3, x4])
    x = Reshape((4, feat_model.output_shape[1], 1), name='reshape1')(x)

    # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
    x = Conv2D(mid_compare_dim, (4, 1), activation='sigmoid', padding='valid')(x)
    x = Reshape((feat_model.output_shape[1], mid_compare_dim, 1))(x)
    x = Conv2D(1, (1, mid_compare_dim), activation='sigmoid', padding='valid')(x)
    x = Flatten(name='flatten')(x)

    # Weighted sum implemented as a Dense layer.
    x = Dense(1, use_bias=True, activation='sigmoid', name='weighted_average')(x)
    comp_model = keras.Model([xa_inp, xb_inp], x, name='compare')

    # simaese model
    im_a = Input(shape=(img_shape[0], img_shape[1], 3))
    im_b = Input(shape=(img_shape[0], img_shape[1], 3))
    im_a_feat = feat_model(im_a)
    im_b_feat = feat_model(im_b)
    y_out = comp_model([im_a_feat, im_b_feat])
    model = keras.Model([im_a, im_b], y_out)

    return model, feat_model, comp_model
