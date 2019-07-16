#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : backend.py
#   Author      : YunYang1994
#   Created date: 2019-07-16 00:24:11
#   Description :
#
#================================================================

import numpy as np
import tensorflow as tf


weights = np.load("./vgg16.npy", encoding='latin1').item()


inputs = tf.keras.layers.Input([224, 224, 3])

# Block 1
conv1_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=[1, 1],
                              padding='same', activation='relu', use_bias=True, name='conv1_1')(inputs)
conv1_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=[1, 1],
                              padding='same', activation='relu', use_bias=True, name='conv1_2')(conv1_1)
pool1_1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1_1')

# Block 2
conv2_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=[1, 1],
                              padding='same', activation='relu', use_bias=True, name='conv2_1')(pool1_1)
conv2_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=[1, 1],
                              padding='same', activation='relu', use_bias=True, name='conv2_2')(conv2_1)
pool2_1 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2_1')

# Block 3
conv3_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=[1, 1],
                              padding='same', activation='relu', use_bias=True, name='conv3_1')(pool2_1)
conv3_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=[1, 1],
                              padding='same', activation='relu', use_bias=True, name='conv3_2')(conv3_1)
conv3_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=[1, 1],
                              padding='same', activation='relu', use_bias=True, name='conv3_3')(conv3_2)
pool3_1 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3_1')

# Block 4
conv4_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=[1, 1],
                              padding='same', activation='relu', use_bias=True, name='conv4_1')(pool3_1)
conv4_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=[1, 1],
                              padding='same', activation='relu', use_bias=True, name='conv4_2')(conv4_1)
conv4_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=[1, 1],
                              padding='same', activation='relu', use_bias=True, name='conv4_3')(conv4_2)
pool4_1 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4_1')

# Block 5
conv5_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=[1, 1],
                              padding='same', activation='relu', use_bias=True, name='conv5_1')(pool4_1)
conv5_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=[1, 1],
                              padding='same', activation='relu', use_bias=True, name='conv5_2')(conv5_1)
conv5_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=[1, 1],
                              padding='same', activation='relu', use_bias=True, name='conv5_3')(conv5_2)





model = tf.keras.Model(inputs, conv)




