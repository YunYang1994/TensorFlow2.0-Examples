#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : ssd.py
#   Author      : YunYang1994
#   Created date: 2019-11-06 16:56:46
#   Description :
#
#================================================================

import tensorflow as tf
from core.backbone import vgg16

class SSD(tf.keras.Model):
    def __init__(self, input_data, num_class=21):
        super(SSD, self).__init__()
        # conv1
        conv4, conv = vgg16(input_data)
        self.conv4 = tf.keras.layers.Conv2D(4*(num_class + 5),3, padding='same')(conv4)
        # fc6, from now they are all SSD blocks
        conv = tf.keras.layers.Conv2D(1024, 3, dilation_rate=6, activation='relu', padding='same')(conv)#fc6
        # fc7
        conv = tf.keras.layers.Conv2D(1024, 1, activation='relu', padding='same')(conv)#fc7
        self.conv7 = tf.keras.layers.Conv2D(6*(num_class + 5), 3, padding='same')(conv)
        # Block 8/9/10/11: 1x1 and 3x3 convolutions strides 2 (except the last 2 layers)
        # conv8
        conv = tf.keras.layers.Conv2D(256, 1, activation='relu', padding='same')(conv)
        conv = tf.keras.layers.Conv2D(512, 3, strides=2, activation='relu', padding='same')(conv)
        self.conv8 = tf.keras.layers.Conv2D(6*(num_class + 5),3, padding='same')(conv)
        # conv9
        conv = tf.keras.layers.Conv2D(128, 1, activation='relu', padding='same')(conv)
        conv = tf.keras.layers.Conv2D(256, 3, strides=2, activation='relu', padding='same')(conv)
        self.conv9 = tf.keras.layers.Conv2D(6*(num_class + 5),3, padding='same')(conv)
        # conv10
        conv = tf.keras.layers.Conv2D(128, 1, activation='relu', padding='same')(conv)
        conv = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='valid')(conv)
        self.conv10 = tf.keras.layers.Conv2D(4*(num_class + 5),3, padding='same')(conv)
        # conv11
        conv = tf.keras.layers.Conv2D(128, 1, activation='relu', padding='same')(conv)
        conv = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='valid')(conv)
        self.conv11 = tf.keras.layers.Conv2D(4*(num_class + 5),3, padding='same')(conv)

    def display(self):
        print(self.conv4.shape)
        print(self.conv7.shape)
        print(self.conv8.shape)
        print(self.conv9.shape)
        print(self.conv10.shape)
        print(self.conv11.shape)
        return self.conv4, self.conv7, self.conv8, self.conv9, self.conv10, self.conv11

model = SSD(tf.ones(shape=[1,300,300,3]),21)
model.display()





