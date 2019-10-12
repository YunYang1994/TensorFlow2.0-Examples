#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-10-12 17:44:30
#   Description :
#
#================================================================

import cv2
import tensorflow as tf
from config import colormap
import numpy as np

def preprocessing(image_path, label_path):
    image_path, label_path = tf.strings.strip(image_path), tf.strings.strip(label_path)
    image = tf.image.decode_jpeg(tf.io.read_file(image_path)) # RGB channels
    label = tf.image.decode_png(tf.io.read_file(label_path))
    image = tf.image.resize(image, [224, 224], method='nearest')
    label = tf.image.resize(label, [224, 224], method='nearest')

    new_label = np.zeros(shape=[224, 224, 21])
    for i in range(224):
        for j in range(224):
            pix_color = label[i][j].numpy().tolist()
            print(pix_color)
            if pix_color in colormap:
                cls_idx = colormap.index(pix_color)
                new_label[i][j][cls_idx] = 1.
    return image, new_label*255


# def preprocessing(image_path, label_path):
    # image = cv2.imread(image_path) # RGB channels
    # print(image.shape)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # label = cv2.imread(label_path) # RGB channels
    # image = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

    # H, W, C = label.shape
    # print(label.shape)
    # new_label = np.zeros(shape=[H, W, 21])
    # for i in range(H):
        # for j in range(W):
            # pix_color = label[i][j].tolist()
            # print(pix_color)
            # if pix_color in colormap:
                # cls_idx = colormap.index(pix_color)
                # new_label[i][j][cls_idx] = 1.
    # return image, new_label*255

image_paths = open("./data/train_image.txt").readlines()
label_paths = open("./data/train_label.txt").readlines()

dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
dataset = dataset.map(preprocessing).batch(1).repeat(5).shuffle(4)
TrainGenerator = iter(dataset)

image_path, label_path = next(TrainGenerator)
# image_path = image_path[0].numpy().rstrip()
# label_path = label_path[0].numpy().rstrip()
# image, label = preprocessing(image_path, label_path)



# cv2.imwrite("label.png", label[0].numpy())
# data=cv2.cvtColor(image[0].numpy(), cv2.COLOR_BGR2RGB)
# cv2.imwrite("image.jpg", data)
