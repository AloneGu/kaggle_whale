#!/usr/bin/env python
# encoding: utf-8

"""
@author: Jackling Gu
@file: gen_data_dict.py
@time: 2/13/19 20:57
"""

# Image,Id
# 0000e88ab.jpg,w_f48451c
# 0001f9222.jpg,w_c3d896a

import pandas as pd
import os
import json

BASE_IMG_DIR = '../data/train'

# read data
df = pd.read_csv('../data/train.csv')
cls_fp_d = {}

for f, cls_name in df.values:
    fp = os.path.join(BASE_IMG_DIR, f)
    if cls_name not in cls_fp_d:
        cls_fp_d[cls_name] = []
    cls_fp_d[cls_name].append(fp)

with open('../data/train_data.json', 'w') as fout:
    json.dump(cls_fp_d, fout, indent=2)

print('label cnt', len(cls_fp_d))

label_to_id = {}
id_to_label = {}
for i, k in enumerate(sorted(list(cls_fp_d.keys()))):
    label_to_id[k] = i
    id_to_label[i] = k

with open('../data/id_to_label.json', 'w') as fout:
    json.dump(id_to_label, fout, indent=2)

with open('../data/label_to_id.json', 'w') as fout:
    json.dump(label_to_id, fout, indent=2)

print('save done')
