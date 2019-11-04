#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-11-04 13:46:45
#   Description :
#
#================================================================

import os
import cv2
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from loss import compute_loss
from MonodepthNetwork import MonodepthNetwork


def create_path_generator(image_paths_txt):
    lines = open(image_paths_txt).readlines()
    left_right_image_paths = []
    for line in lines:
        left_image_path, right_image_path = line.split(" ")
        left_right_image_paths.append((left_image_path.rstrip(), right_image_path.rstrip()))
    while True:
        random.shuffle(left_right_image_paths)
        for i in range(len(left_right_image_paths)):
            yield left_right_image_paths[i]

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_NEAREST)
    return image

def DataGenerator(image_paths_txt, batch_size=1):
    data_dir_path = os.path.dirname(image_paths_txt)
    path_generator = create_path_generator(image_paths_txt)
    while True:
        left_images  = np.zeros(shape=[batch_size, 256, 512, 3])
        right_images = np.zeros(shape=[batch_size, 256, 512, 3])

        for i in range(batch_size):
            left_image_path, right_image_path = next(path_generator)
            left_image_path = data_dir_path + "/" + left_image_path
            right_image_path = data_dir_path + "/" + right_image_path

            left_image = cv2.imread(left_image_path)
            left_image = preprocess(left_image)

            right_image = cv2.imread(right_image_path)
            right_image = preprocess(right_image)

            left_images[i]  = left_image
            right_images[i] = right_image

        yield left_images.astype(np.float32), right_images.astype(np.float32)

epochs = 50
model = MonodepthNetwork()
optimizer = tf.keras.optimizers.Adam()
image_paths_txt = "/media/yang/e2053f1d-2479-4407-a2f3-7d7c3bfd5f9c/kitti_raw/kitti_train_files.txt"
writer = tf.summary.create_file_writer("./log")
trainset = DataGenerator(image_paths_txt, 4)

for epoch in range(epochs):
    for step in range(5000):
        with tf.GradientTape() as tape:
            left_images, right_images = next(trainset)
            lr_disp = model(left_images, training=True)
            image_loss, disp_gradient_loss, lr_loss = compute_loss(left_images, right_images, lr_disp)
            total_loss = image_loss + disp_gradient_loss + lr_loss
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print("EPOCH %2d STEP %3d total_loss %.6f image_loss %.6f disp_gradient_loss %.6f lr_loss %.6f" %(
                    epoch, step, total_loss.numpy(), image_loss.numpy(), disp_gradient_loss.numpy(), lr_loss.numpy()))

            # writing summary data
            global_steps = step + epoch * 5000
            with writer.as_default():
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/image_loss", image_loss, step=global_steps)
                tf.summary.scalar("loss/gradient_loss", disp_gradient_loss, step=global_steps)
                tf.summary.scalar("loss/lr_loss", lr_loss, step=global_steps)
            writer.flush()



