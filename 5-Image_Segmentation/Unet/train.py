#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-09-19 15:25:10
#   Description :
#
#================================================================

import os
import cv2
import json
import numpy as np
import tensorflow as tf
from model import Unet

image_size = 256
epochs = 1
batch_size = 2
model = Unet(21, image_size)

image_paths = open("./data/train_image.txt").readlines()
label_paths = open("./data/train_label.txt").readlines()
voc_colormap = json.load(open("./data/voc_colormap.json"))

class_name = sorted(voc_colormap.keys())
colormap = [voc_colormap[cls] for cls in class_name]



for epoch in range(epochs):
    batch_image = np.zeros(shape=[batch_size, image_size, image_size, 3], dtype=np.float32)
    batch_label = np.zeros(shape=[batch_size, image_size, image_size, 21], dtype=np.float32)
    for i in range(batch_size):
        image = cv2.imread(image_paths.pop().rstrip())
        image = cv2.resize(image, dsize=(image_size, image_size), interpolation=cv2.INTER_NEAREST)
        batch_image[i] = image

        label_image = cv2.imread(label_paths.pop().rstrip())
        label_image = cv2.resize(label_image, dsize=(image_size, image_size), interpolation=cv2.INTER_NEAREST)

        H,W,C = label_image.shape
        for x in range(H):
            for y in range(W):
                pixel_color = image[x][y].tolist()
                if pixel_color in colormap:
                    cls_idx = colormap.index(pixel_color)
                    batch_label[i][x][y][cls_idx] = 1.

    with tf.GradientTape() as tape:
        pred_result = model(batch_image, training=True)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_result, batch_label))
        print(loss)








