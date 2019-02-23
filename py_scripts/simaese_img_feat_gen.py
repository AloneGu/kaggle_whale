#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: simaese_img_feat_gen.py
@time: 2/23/19 20:01
"""

import os
import pickle
import json
import argparse
from keras.models import load_model
from gen_model_pred import predict_one_img

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--model_path')
    args.add_argument('--output_path', default=None)

    opts = args.parse_args()
    print(opts)

    model_name = os.path.basename(opts.model_path[:-3])
    if opts.output_path is None:
        output_p = '../data/simaese_img_feats/simaese_{}.pkl'.format(model_name)
    else:
        output_p = opts.output_path
    print('out', output_p)

    sim_model = load_model(opts.model_path, compile=False)
    feat_model = sim_model.layers[2]
    img_shape = feat_model.input_shape[1:3]
    print(img_shape)

    # for each image in train
    all_data = json.loads(open('../data/train_data.json').read())
    all_data.pop('new_whale')
    output_feat = []  # save label, image path, image feat
    p_cnt = 0
    for label in all_data:
        for fp in all_data[label]:
            img_feat = predict_one_img(feat_model, fp, img_shape)
            output_feat.append([label, fp, img_feat])
            p_cnt += 1
            if p_cnt % 1000 == 5:
                print(p_cnt, label, fp, img_feat[:5])

    with open(output_p, 'wb') as fout:
        pickle.dump(output_feat, fout)
