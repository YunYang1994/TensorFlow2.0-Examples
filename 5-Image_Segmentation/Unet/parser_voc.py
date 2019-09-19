#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : parser_voc.py
#   Author      : YunYang1994
#   Created date: 2019-09-19 15:53:00
#   Description :
#
#================================================================

import os

VOC_path = "/home/yang/dataset/VOC"

train_label_write = open(os.path.join(os.getcwd(), "data/train_label.txt"), "a+")
train_image_write = open(os.path.join(os.getcwd(), "data/train_image.txt"), "a+")
test_label_write = open(os.path.join(os.getcwd(), "data/test_label.txt"), "a+")
test_image_write = open(os.path.join(os.getcwd(), "data/test_image.txt"), "a+")

for mode in ["train", "test"]:
    for year in [2007, 2012]:
        if mode == "test" and year == 2012: continue
        train_label_folder = os.path.join(VOC_path, "%s/VOCdevkit/VOC%d/SegmentationClass" %(mode, year))
        train_image_folder = os.path.join(VOC_path, "%s/VOCdevkit/VOC%d/JPEGImages" %(mode, year))
        train_label_images = os.listdir(train_label_folder)

        for train_label_image in train_label_images:
            label_name = train_label_image[:-4]
            image_path = os.path.join(train_image_folder, label_name + ".jpg")
            if not os.path.exists(image_path): continue
            label_path = os.path.join(train_label_folder, train_label_image)
            train_label_write.writelines(label_path+"\n")
            train_image_write.writelines(image_path+"\n")


