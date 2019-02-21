#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: gen_model_pred.py
@time: 2/20/19 20:14
"""

import argparse
import pickle
import datetime
import numpy as np
import pandas as pd

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--pred_path')
    args.add_argument('--binary_path', default=None, type=str)
    args.add_argument('--binary_r', default=1.0, type=float)
    args.add_argument('--thres', default=0.4, type=float)
    args.add_argument('--output_path', default=None)
    opts = args.parse_args()
    print(opts)

    fin = open(opts.pred_path, 'rb')
    res = pickle.load(fin)
    fin.close()

    fl = []
    pred_probs = []
    labels = []
    label_cvt_list = ['new_whale'] + res['label_list']
    for k in res['pred']:
        tmp_pred = res['pred'][k]
        pred_probs.append(tmp_pred)
        fl.append(k)
    # insert new_whale prob
    img_cnt = len(fl)

    if opts.binary_path is None:  # simple insert
        new_whale_prob = np.ones(shape=(img_cnt, 1)) * opts.thres
    else:
        with open(opts.binary_path, 'rb') as fin:
            binary_res = pickle.load(fin)
            binary_pred = binary_res['pred']
        new_whale_prob = []
        for k in fl:
            new_whale_prob.append(binary_pred[k][0])  # first idx is for new whale
        new_whale_prob = np.array(new_whale_prob) * opts.binary_r
        new_whale_prob = np.expand_dims(new_whale_prob, -1)
        print('binary new whale', new_whale_prob[:10])

    final_pred_prob = np.hstack((new_whale_prob, np.array(pred_probs)))
    print(final_pred_prob.shape)
    for i in range(img_cnt):
        tmp_labels = final_pred_prob[i].argsort()[-5:][::-1]
        tmp_labels = [label_cvt_list[j] for j in tmp_labels]
        tmp_label_str = ' '.join(tmp_labels)
        labels.append(tmp_label_str)
        if i % 1000 == 5:
            print(i, fl[i], tmp_label_str)

    df = pd.DataFrame({'Image': fl, 'Id': labels})
    tmp_time_str = str(datetime.datetime.now())[:16]
    tmp_time_str = tmp_time_str.replace(' ', '_').replace(':', '_')
    if opts.output_path is None:
        output_p = '../submissions/sub_{}.csv'.format(tmp_time_str)
    else:
        output_p = opts.output_path
    df.to_csv(output_p, header=True, index=False)
    print(df.head(10))
    print('done')
