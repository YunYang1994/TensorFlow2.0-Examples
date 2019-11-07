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


class SSD(tf.keras.Model):
    def __init__(self, num_class=21):
        super(SSD, self).__init__()
        # conv1
        self.conv1_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.conv1_2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.pool1   = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')

        # conv2
        self.conv2_1 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        self.conv2_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        self.pool2   = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')

        # conv3
        self.conv3_1 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.conv3_2 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.conv3_3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.pool3   = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')

        # conv4
        self.conv4_1 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.conv4_2 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.conv4_3 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.pool4   = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')

        # conv5
        self.conv5_1 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.conv5_2 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.conv5_3 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.pool5   = tf.keras.layers.MaxPooling2D(3, strides=1, padding='same')

        # fc6, => vgg backbone is finished. now they are all SSD blocks
        self.fc6 = tf.keras.layers.Conv2D(1024, 3, dilation_rate=6, activation='relu', padding='same')
        # fc7
        self.fc7 = tf.keras.layers.Conv2D(1024, 1, activation='relu', padding='same')
        # Block 8/9/10/11: 1x1 and 3x3 convolutions strides 2 (except lasts)
        # conv8
        self.conv8_1 = tf.keras.layers.Conv2D(256, 1, activation='relu', padding='same')
        self.conv8_2 = tf.keras.layers.Conv2D(512, 3, strides=2, activation='relu', padding='same')
        # conv9
        self.conv9_1 = tf.keras.layers.Conv2D(128, 1, activation='relu', padding='same')
        self.conv9_2 = tf.keras.layers.Conv2D(256, 3, strides=2, activation='relu', padding='same')
        # conv10
        self.conv10_1 = tf.keras.layers.Conv2D(128, 1, activation='relu', padding='same')
        self.conv10_2 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='valid')
        # conv11
        self.conv11_1 = tf.keras.layers.Conv2D(128, 1, activation='relu', padding='same')
        self.conv11_2 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='valid')



    def call(self, x, training=False):
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h = self.pool1(h)

        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h = self.pool2(h)

        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.conv3_3(h)
        h = self.pool3(h)

        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv4_3(h)
        print(h.shape)
        h = self.pool4(h)

        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = self.conv5_3(h)
        h = self.pool5(h)

        h = self.fc6(h)     # [1,19,19,1024]
        h = self.fc7(h)     # [1,19,19,1024]
        print(h.shape)

        h = self.conv8_1(h)
        h = self.conv8_2(h) # [1,10,10, 512]
        print(h.shape)

        h = self.conv9_1(h)
        h = self.conv9_2(h) # [1, 5, 5, 256]
        print(h.shape)

        h = self.conv10_1(h)
        h = self.conv10_2(h) # [1, 3, 3, 256]
        print(h.shape)

        h = self.conv11_1(h)
        h = self.conv11_2(h) # [1, 1, 1, 256]
        print(h.shape)
        return h

model = SSD(21)
x = model(tf.ones(shape=[1,300,300,3]))





