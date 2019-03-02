#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: simaese_gen_sub.py
@time: 2/23/19 20:32
"""

# get img feats for test images
# load train img feats
# get sim scores and save in 5004 dim
# insert new_whale score using a threshold
# sort and gen submission


import os
import glob
import pickle
import json
import argparse
from keras.models import load_model
import datetime
import numpy as np
import pandas as pd
from gen_model_pred import predict_one_img

SPEED_UP_FLAG = False
MAX_SAMPLE = 10
FILTER_TOP3 = True
FILTER_TOP3_LABELS = json.loads(open('../data/top3_filter.json').read())
USING_MEAN_SIM = False


def get_test_img_feat(img_model, img_shape):
    test_fl = list(glob.glob('../data/test/*'))
    test_fl = sorted(test_fl)
    res = []
    p_cnt = 0
    for f in test_fl:  # test
        tmp_feat = predict_one_img(img_model, f, img_shape)
        base_fn = os.path.basename(f)
        res.append([base_fn, tmp_feat])
        p_cnt += 1
        if p_cnt % 1000 == 5:
            print(p_cnt, f, tmp_feat[:3])
    return res


def get_sim_score(comp_model, train_feats, test_feats, label_cvt):
    sim_res = {}
    # x: label, fp, feat
    pred_train_feat = [x[2] for x in train_feats]
    pred_train_feat = np.array(pred_train_feat)
    train_cnt = len(train_feats)
    p_cnt = 0
    for k, test_f in test_feats:
        # pred for all pairs
        dup_test_feats = np.tile(test_f, (train_cnt, 1))
        scores = comp_model.predict([pred_train_feat, dup_test_feats])
        # print(scores.shape)

        tmp_pred = [0 for i in range(5005)]
        # keep highest score
        if USING_MEAN_SIM is False:
            for i, tmp_score in enumerate(scores):
                label = train_feats[i][0]
                label_idx = label_cvt[label]
                tmp_pred[label_idx] = max(tmp_pred[label_idx], tmp_score[0])  # keep highest similarity
        else:
            # using mean score
            tmp_score_list = [[] for i in range(5005)]
            for i, tmp_score in enumerate(scores):
                label = train_feats[i][0]
                label_idx = label_cvt[label]
                tmp_score_list[label_idx].append(tmp_score[0])
            for i, score_l in enumerate(tmp_score_list):
                if len(score_l) > 0:
                    tmp_pred[i] = np.mean(score_l)  # using mean sim score

        sim_res[k] = tmp_pred
        p_cnt += 1
        if p_cnt % 200 == 5:
            print(p_cnt, 'done', k, tmp_pred[:3], datetime.datetime.now())
    return sim_res


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--model_path')
    args.add_argument('--train_feat')
    args.add_argument('--output_path', default=None)
    args.add_argument('--thres', default=0.99, type=float)
    args.add_argument('--test_cnt', default=1000000, type=int)

    opts = args.parse_args()
    print(opts)

    model_name = os.path.basename(opts.model_path[:-3])
    if opts.output_path is None:
        tmp_time_str = str(datetime.datetime.now())[:16]
        tmp_time_str = tmp_time_str.replace(' ', '_').replace(':', '_')
        output_p = '../submissions/simaese_{}_{}.csv'.format(model_name, tmp_time_str)
    else:
        output_p = opts.output_path
    print('out', output_p)

    sim_model = load_model(opts.model_path, compile=False)
    feat_model = sim_model.layers[2]
    comp_model = sim_model.layers[3]
    img_shape = feat_model.input_shape[1:3]
    # comp_model.summary()
    print('img shape', img_shape)

    # get test images feat
    test_feats = get_test_img_feat(feat_model, img_shape)
    print('test feats done', len(test_feats))
    if opts.test_cnt is not None:
        test_feats = test_feats[:opts.test_cnt]

    # build id label info
    label_to_id = json.loads(open('../data/label_to_id.json').read())
    label_cvt_list = ['' for i in range(5005)]
    for l in label_to_id:
        idx = label_to_id[l]
        idx = int(idx)
        label_cvt_list[idx] = l

    # get sim score for each label, key:5005 score
    fin = open(opts.train_feat, 'rb')
    train_feats = pickle.load(fin)
    fin.close()
    print('train feat cnt', len(train_feats))

    # bad idea to keep topk samples
    if SPEED_UP_FLAG:
        tmp_train_feats = []
        train_cnt_d = {}
        for label, fp, tmp_f in train_feats:
            if label not in train_cnt_d:
                train_cnt_d[label] = 0
            train_cnt_d[label] += 1
            if train_cnt_d[label] < MAX_SAMPLE:
                tmp_train_feats.append([label, fp, tmp_f])
        train_feats = tmp_train_feats
        print('filtered train feat cnt', len(train_feats))

    # filter based on public top3 labels
    if FILTER_TOP3:
        tmp_train_feats = []
        train_cnt_d = {}
        for label, fp, tmp_f in train_feats:
            if label in FILTER_TOP3_LABELS:  # only keep labels in top3 set
                tmp_train_feats.append([label, fp, tmp_f])
        train_feats = tmp_train_feats
        print('filtered train feat cnt', len(train_feats))

    sim_res = get_sim_score(comp_model, train_feats, test_feats, label_to_id)
    print('sim res done')

    # insert new whale and gen sub
    fl, labels = [], []
    for k in sim_res:
        fl.append(k)
        pred = sim_res[k]
        pred[0] = opts.thres  # replace new_whale score
        pred = np.array(pred)
        tmp_labels = pred.argsort()[-5:][::-1]
        tmp_labels = [label_cvt_list[j] for j in tmp_labels]
        tmp_label_str = ' '.join(tmp_labels)
        labels.append(tmp_label_str)
    df = pd.DataFrame({'Image': fl, 'Id': labels})
    df.to_csv(output_p, header=True, index=False)
    print(df.head(10))
    print('done')
