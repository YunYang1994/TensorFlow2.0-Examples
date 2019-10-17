#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : parser_voc.py
#   Author      : YunYang1994
#   Created date: 2019-10-12 17:50:18
#   Description :
#
#================================================================

import os
import argparse
import numpy as np
from config import colormap
from scipy import misc

if not os.path.exists("./data"): os.mkdir("./data")
if not os.path.exists("./data/train_labels"): os.mkdir("./data/train_labels")
if not os.path.exists("./data/test_labels"): os.mkdir("./data/test_labels")

parser = argparse.ArgumentParser()
parser.add_argument("--voc_path", type=str, default="/home/yang/dataset/VOC")
flags = parser.parse_args()
if not os.path.exists(flags.voc_path): # "/home/yang/dataset/VOC"
    raise ValueError("Path: %s does not exist" %flags.voc_path)

for mode in ["train", "test"]:
    image_write = open(os.path.join(os.getcwd(), "data/%s_image.txt" %mode), "w")
    for year in [2007, 2012]:
        if mode == "test" and year == 2012: continue
        train_label_folder = os.path.join(flags.voc_path, "%s/VOCdevkit/VOC%d/SegmentationClass" %(mode, year))
        train_image_folder = os.path.join(flags.voc_path, "%s/VOCdevkit/VOC%d/JPEGImages" %(mode, year))
        train_label_images = os.listdir(train_label_folder)

        for train_label_image in train_label_images:
            label_name = train_label_image[:-4]
            image_path = os.path.join(train_image_folder, label_name + ".jpg")
            if not os.path.exists(image_path): continue
            image_write.writelines(image_path+"\n")
            label_path = os.path.join(train_label_folder, train_label_image)
            label_image = np.array(misc.imread(label_path))
            write_label = open(("./data/%s_labels/" % mode)+label_name+".txt", 'w')
            print("=> processing %s" %label_path)
            H, W, C = label_image.shape
            for i in range(H):
                write_line = []
                for j in range(W):
                    pixel_color = label_image[i, j].tolist()
                    if pixel_color in colormap:
                        cls_idx = colormap.index(pixel_color)
                    else:
                        cls_idx = 0
                    write_line.append(str(cls_idx))
                write_label.writelines(" ".join(write_line) + "\n")

