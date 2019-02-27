#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: gen_model_pred.py
@time: 2/20/19 20:14
"""
import os
import sys
from keras.models import load_model
import argparse
import json
import pickle
import glob
import numpy as np
from utils import preprocess_func, get_bbox, expand_bb
from keras.preprocessing.image import load_img, img_to_array

ID_TO_LABEL = json.loads(open('../data/id_to_label.json').read())
TEST_FILE_LIST = glob.glob('../data/test/*')


# pred with box and flip lr
def predict_one_img(model, img_path, img_shape):
    h, w = img_shape
    img = load_img(img_path)
    # crop
    base_fn = os.path.basename(img_path)
    box = get_bbox(base_fn)
    x0, y0, x1, y1 = expand_bb(img, box)
    if not (x0 >= x1 or y0 >= y1):
        tmp_box = (x0, y0, x1, y1)
        img.crop(tmp_box)
    # pil use w,h
    img = img.resize((w, h))  # default nearest
    img = img_to_array(img)
    img = preprocess_func(img)
    flip_img = np.fliplr(img)
    res = model.predict(np.array([img, flip_img]))
    res = np.mean(res, axis=0)
    return res


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--model_path')
    args.add_argument('--cls_num', default=1, type=int)
    args.add_argument('--output_path', default=None)

    opts = args.parse_args()
    print(opts)
    model = load_model(opts.model_path, compile=False)
    image_shape = model.input_shape[1:3]  # None,224,224,3
    print('image shape', image_shape)

    res = {}
    res['label_list'] = []
    res['pred'] = {}
    if opts.cls_num == 2:
        res['label_list'] = ['new_whale', 'not_new_whale']
    elif opts.cls_num == 5005:
        for i in range(5005):
            res['label_list'].append(ID_TO_LABEL[str(i)])
    elif opts.cls_num == 5004:
        for i in range(5004):
            res['label_list'].append(ID_TO_LABEL[str(i + 1)])
    else:
        print('wrong conf')
        sys.exit()

    print('start prediction')
    for i, fp in enumerate(TEST_FILE_LIST):
        key = os.path.basename(fp)
        pred = predict_one_img(model, fp, image_shape)
        res['pred'][key] = pred
        if i % 500 == 5:
            print(i, key, pred[:10])

    # save
    if opts.output_path is None:
        model_name = os.path.basename(opts.model_path)[:-3]
        output_p = '../data/save_feats/{}_{}_feat.pkl'.format(model_name, image_shape[0])
    else:
        output_p = opts.output_path
    if not os.path.exists(os.path.dirname(output_p)):
        os.makedirs(os.path.dirname(output_p))
    with open(output_p, 'wb') as fout:
        pickle.dump(res, fout)
    print('save done', output_p)
