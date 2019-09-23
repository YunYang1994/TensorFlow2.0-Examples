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
import shutil
import numpy as np
import tensorflow as tf
from Unet import Unet

image_size = 512
epochs = 50
lr = 0.0005
batch_size = 2
model = Unet(21, image_size)
logdir = "./log"
global_steps = 0
optimizer = tf.keras.optimizers.Adam(lr)

voc_colormap = json.load(open("./data/voc_colormap.json"))

if os.path.exists(logdir): shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)

class_name = sorted(voc_colormap.keys())
colormap = [voc_colormap[cls] for cls in class_name]

for epoch in range(epochs):
    image_paths = open("./data/train_image.txt").readlines()
    label_paths = open("./data/train_label.txt").readlines()
    batch_image = np.zeros(shape=[batch_size, image_size, image_size, 3], dtype=np.float32)
    batch_label = np.zeros(shape=[batch_size, image_size, image_size, 21], dtype=np.float32)
    while len(image_paths):
        global_steps += 1
        for i in range(batch_size):
            image = cv2.imread(image_paths.pop().rstrip())
            image = cv2.resize(image, dsize=(image_size, image_size), interpolation=cv2.INTER_NEAREST)
            batch_image[i] = image / 255.

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
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred_result, labels=batch_label)
            loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1,2]))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            # writing summary data
        with writer.as_default():
            tf.summary.scalar("loss", loss, step=global_steps)
            print("=> Epoch: %2d, global_steps: %5d loss: %.6f" %(epoch+1, global_steps, loss.numpy()))
        writer.flush()
    model.save_weights("Unet.h5")



