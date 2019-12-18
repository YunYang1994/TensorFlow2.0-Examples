#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : utils.py
#   Author      : YunYang1994
#   Created date: 2019-10-12 17:47:24
#   Description :
#
#================================================================

import os
import cv2
import random
import numpy as np

from PIL import Image

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

rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])

def visual_result(image, label, alpha=0.7):
    """
    image shape -> [H, W, C]
    label shape -> [H, W]
    """
    image = (image * rgb_std + rgb_mean) * 255
    image, label = image.astype(np.int), label.astype(np.int)
    H, W, C = image.shape
    masks_color = np.zeros(shape=[H, W, C])
    inv_masks_color = np.zeros(shape=[H, W, C])
    cls = []
    for i in range(H):
        for j in range(W):
            cls_idx = label[i, j]
            masks_color[i, j] = np.array(colormap[cls_idx])
            cls.append(cls_idx)
            if classes[cls_idx] == "background":
                inv_masks_color[i, j] = alpha * image[i, j]

    show_image = np.zeros(shape=[224, 672, 3])
    cls = set(cls)
    for x in cls:
        print("=> ", classes[x])
    show_image[:, :224, :] = image
    show_image[:, 224:448, :] = masks_color
    show_image[:, 448:, :] = (1-alpha)*image + alpha*masks_color + inv_masks_color
    show_image = Image.fromarray(np.uint8(show_image))
    return show_image

def create_image_label_path_generator(images_filepath, labels_filepath):
    image_paths = open(images_filepath).readlines()
    all_label_txts = os.listdir(labels_filepath)
    image_label_paths = []
    for label_txt in all_label_txts:
        label_name = label_txt[:-4]
        label_path = labels_filepath + "/" + label_txt
        for image_path in image_paths:
            image_path = image_path.rstrip()
            image_name = image_path.split("/")[-1][:-4]
            if label_name == image_name:
                image_label_paths.append((image_path, label_path))
    while True:
        random.shuffle(image_label_paths)
        for i in range(len(image_label_paths)):
            yield image_label_paths[i]

def process_image_label(image_path, label_path):
    # image = misc.imread(image_path)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_NEAREST)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # data augmentation here
    # randomly shift gamma
    gamma = random.uniform(0.8, 1.2)
    image = image.copy() ** gamma
    image = np.clip(image, 0, 255)
    # randomly shift brightness
    brightness = random.uniform(0.5, 2.0)
    image = image.copy() * brightness
    image = np.clip(image, 0, 255)
    # image transformation here
    image = (image / 255. - rgb_mean) / rgb_std

    label = open(label_path).readlines()
    label = [np.array(line.rstrip().split(" ")) for line in label]
    label = np.array(label, dtype=np.int)
    label = cv2.resize(label, (224, 224), interpolation=cv2.INTER_NEAREST)
    label = label.astype(np.int)

    return image, label


def DataGenerator(train_image_txt, train_labels_dir, batch_size):
    """
    generate image and mask at the same time
    """
    image_label_path_generator = create_image_label_path_generator(
        train_image_txt, train_labels_dir
    )
    while True:
        images = np.zeros(shape=[batch_size, 224, 224, 3])
        labels = np.zeros(shape=[batch_size, 224, 224], dtype=np.float)
        for i in range(batch_size):
            image_path, label_path = next(image_label_path_generator)
            image, label = process_image_label(image_path, label_path)
            images[i], labels[i] = image, label
        yield images, labels
