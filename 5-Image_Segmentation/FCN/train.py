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

import os
import cv2
import random
import tensorflow as tf
from config import colormap, classes
import numpy as np
from PIL import Image
from scipy import misc



def create_image_label_path_generator(images_filepath, labels_filepath):
    image_paths = open(images_filepath).readlines()
    all_label_txts = os.listdir(labels_filepath)
    print(all_label_txts)
    image_label_paths = []
    for label_txt in all_label_txts:
        label_name = label_txt[:-4]
        label_path = labels_filepath + "/" + label_txt
        for image_path in image_paths:
            image_path = image_path.rstrip()
            image_name = image_path.split("/")[-1][:-4]
            if label_name == image_name:
                image_label_paths.append((image_path, label_path))
    print(image_label_paths)
    while True:
        random.shuffle(image_label_paths)
        for i in range(len(image_label_paths)):
            yield image_label_paths[i]

image_label_path_generator = create_image_label_path_generator(
    "./data/train_image.txt", "./data/train_labels"
)

def process_image_label(image_path, label_path):
    # image = misc.imread(image_path)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_NEAREST)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)

    label = open(label_path).readlines()
    label = [np.array(line.rstrip().split(" ")) for line in label]
    label = np.array(label, dtype=np.int)
    label = cv2.resize(label, (224, 224), interpolation=cv2.INTER_NEAREST)

    return image, label



def TrainGenerator(batch_size):
    """
    generate image and mask at the same time
    """
    pass


for epoch in range(4):
    image_path, label_path = next(image_label_path_generator)
    # print(image_path, label_path)
    image, label = process_image_label(image_path, label_path)

    H, W, C = image.shape
    new_label = np.zeros(shape=[H, W, C])
    cls = []
    for i in range(H):
        for j in range(W):
            new_label[i, j] = np.array(colormap[label[i,j]])
            cls.append(label[i, j])

    show_image = 0.7*new_label + 0.3*image
    write_image = np.zeros(shape=[224, 448, 3])
    write_image[:, :224, :] = image
    write_image[:, 224:, :] = show_image
    cls = set(cls)
    # for x in cls:
        # print(classes[x])
    # misc.imshow(show_image)
    misc.imshow(write_image)
    misc.imsave("%d.jpg"%epoch, write_image)


