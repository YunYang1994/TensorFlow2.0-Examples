#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : test.py
#   Author      : YunYang1994
#   Created date: 2019-09-23 14:57:12
#   Description :
#
#================================================================

import os
import cv2
import json
import numpy as np
import tensorflow as tf
from Unet import Unet


batch_size = 2
image_size = 512
alpha = 0.3
model = Unet(21, image_size)
model.load_weights("Unet.h5")

image_paths = open("./data/test_image.txt").readlines()
label_paths = open("./data/test_label.txt").readlines()
voc_colormap = json.load(open("./data/voc_colormap.json"))
if not os.path.exists("./images"): os.mkdir("./images")

class_name = sorted(voc_colormap.keys())
colormap = [voc_colormap[cls] for cls in class_name]

while len(image_paths):
    output_mask = np.zeros([image_size, image_size, 3], np.uint8)
    image_path = image_paths.pop().rstrip()
    image_name = image_path.split("/")[-1]
    image = cv2.imread(image_path)
    image = cv2.resize(image, dsize=(image_size, image_size), interpolation=cv2.INTER_NEAREST)
    batch_image = np.expand_dims(image / 255., 0)

    pred_mask = model.predict(batch_image)[0]
    pred_mask = tf.nn.softmax(pred_mask, axis=-1)
    pred_mask = tf.argmax(pred_mask, axis=-1).numpy()
    for i in range(image_size):
        for j in range(image_size):
            cls_idx = pred_mask[i][j]
            color = colormap[cls_idx]
            output_mask[i][j] = color
    image = (1-alpha) * output_mask + alpha * image
    cv2.imwrite("./images/"+image_name, image)


