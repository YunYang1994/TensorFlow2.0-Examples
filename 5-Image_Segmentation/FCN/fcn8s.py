#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : fcn8s.py
#   Author      : YunYang1994
#   Created date: 2019-10-12 15:42:20
#   Description :
#
#================================================================

import tensorflow as tf

class FCN8s(tf.keras.Model):
    def __init__(self, n_class=21):
        super(FCN8s, self).__init__()
        # conv1
        self.conv1_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='valid')
        self.conv1_2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.pool1   = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same') # 1/2

        # conv2
        self.conv2_1 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        self.conv2_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        self.pool2   = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same') # 1/4

        # conv3
        self.conv3_1 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.conv3_2 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.conv3_3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.pool3   = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same') # 1/8

        # conv4
        self.conv4_1 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.conv4_2 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.conv4_3 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.pool4   = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same') # 1/16

        # conv5
        self.conv5_1 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.conv5_2 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.conv5_3 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.pool5   = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same') # 1/32

        # fc6
        self.fc6 = tf.keras.layers.Conv2D(4096, 7, activation='relu', padding='valid')
        self.drop6 = tf.keras.layers.Dropout(0.5)

        # fc7
        self.fc7 = tf.keras.layers.Conv2D(4096, 1, activation='relu', padding='valid')
        self.drop7 = tf.keras.layers.Dropout(0.5)

        self.socre_fr = tf.keras.layers.Conv2D(n_class, 1)
        self.score_pool3 = tf.keras.layers.Conv2D(n_class, 1)
        self.score_pool4 = tf.keras.layers.Conv2D(n_class, 1)

        self.upscore2 = tf.keras.layers.Conv2DTranspose(
            n_class,  4, strides=2, padding='valid', use_bias=False)
        self.upscore8 = tf.keras.layers.Conv2DTranspose(
            n_class, 16, strides=8, padding='valid', use_bias=False)
        self.upscore_pool4 = tf.keras.layers.Conv2DTranspose(
            n_class,  4, strides=2, padding='valid', use_bias=False)

    def call(self, x, training=False):
        h = x
        h = self.conv1_1(tf.keras.layers.ZeroPadding2D(padding=(100, 100))(h))
        h = self.conv1_2(h)
        h = self.pool1(h)

        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h = self.pool2(h)

        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.conv3_3(h)
        h = self.pool3(h)
        pool3 = h # 1/8

        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv4_3(h)
        h = self.pool4(h)
        pool4 = h # 1/16
        print(pool4.shape)

        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = self.conv5_3(h)
        h = self.pool5(h)

        h = self.fc6(h)
        h = self.drop6(h, training)

        h = self.fc7(h)
        h = self.drop7(h, training)

        h = self.socre_fr(h)
        h = self.upscore2(h)
        upscore2 = h # 1/16
        # print(upscore2.shape)

        h = self.score_pool4(pool4 * 0.01) # XXX: scaling to train at onece
        h = h[:, 5:5+upscore2.shape[1], 5:5+upscore2.shape[2], :] # channel last
        score_pool4c = h # 1/16

        h = upscore2 + score_pool4c # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h # 1/8

        h = self.score_pool3(pool3 * 0.0001) # XXX: scaling to train at onece
        h = h[:,
              9:9+upscore_pool4.shape[1],
              9:9+upscore_pool4.shape[2], :] # channel last
        score_pool3c = h # 1/8

        h = upscore_pool4 + score_pool3c # 1/8

        h = self.upscore8(h)
        h = h[:, 31:31+x.shape[1], 31:31+x.shape[2], :] # channel last

        return h


