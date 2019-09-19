#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : test.py
#   Author      : YunYang1994
#   Created date: 2019-09-19 16:55:53
#   Description :
#
#================================================================

import json

data = {}
filename="voc_colormap.json"
classes = ['background','aeroplane','bicycle','bird','boat',
                      'bottle','bus','car','cat','chair','cow','diningtable',
                      'dog','horse','motorbike','person','potted plant',
                      'sheep','sofa','train','tv/monitor']
# RGB color for each class
colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]

for i,cls in enumerate(classes):
    data[cls] = colormap[i]

with open(filename,"w") as f_obj:
    json.dump(data,f_obj)

with open("voc_colormap.json") as f:
    data = json.load(f)

print(data['background'][0])
