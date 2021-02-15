#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : backbone.py
#   Author      : YunYang1994
#   Created date: 2019-07-11 23:37:51
#   Description :
#
#================================================================

import tensorflow as tf

def vgg16(input_data):

#======================================VGG16_start===================================================
    # conv1
    conv = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(input_data) #conv1_1
    conv = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv) #conv1_2
    conv   = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')(conv) #pool1

    # conv2
    conv = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv)#conv2_1
    conv = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv)#conv2_2
    conv = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')(conv)#pool2

    # conv3
    conv = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv)#conv3_1
    conv = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv)#conv3_2
    conv = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv)#conv3_3
    conv   = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')(conv)#pool3

    # conv4
    conv = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv)
    conv = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv)
    conv = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv)
    conv4 = conv
    conv = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')(conv)

    # conv5
    conv = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv)
    conv = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv)
    conv = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv)
    conv = tf.keras.layers.MaxPooling2D(3, strides=1, padding='same')(conv)

    return conv4, conv


