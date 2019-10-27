#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : mtcnn.py
#   Author      : YunYang1994
#   Created date: 2019-10-26 19:22:04
#   Description :
#
#================================================================

import tensorflow as tf


class PNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(10, 3, 1, name='conv1')
        self.prelu1 = tf.keras.layers.PReLU(shared_axes=[1,2], name="PReLU1")

        self.conv2 = tf.keras.layers.Conv2D(16, 3, 1, name='conv2')
        self.prelu2 = tf.keras.layers.PReLU(shared_axes=[1,2], name="PReLU2")

        self.conv3 = tf.keras.layers.Conv2D(32, 3, 1, name='conv3')
        self.prelu3 = tf.keras.layers.PReLU(shared_axes=[1,2], name="PReLU3")

        self.conv4_1 = tf.keras.layers.Conv2D(2, 1, 1, name='conv4-1')
        self.conv4_2 = tf.keras.layers.Conv2D(4, 1, 1, name='conv4-2')

    def call(self, x, training=False):
        out = self.prelu1(self.conv1(x))
        out = tf.nn.max_pool2d(out, 2, 2, padding="SAME")
        out = self.prelu2(self.conv2(out))
        out = self.prelu3(self.conv3(out))
        score = tf.nn.softmax(self.conv4_1(out), axis=-1)
        boxes = self.conv4_2(out)
        return boxes, score

class RNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(28, 3, 1, name='conv1')
        self.prelu1 = tf.keras.layers.PReLU(shared_axes=[1,2], name="prelu1")

        self.conv2 = tf.keras.layers.Conv2D(48, 3, 1, name='conv2')
        self.prelu2 = tf.keras.layers.PReLU(shared_axes=[1,2], name="prelu2")

        self.conv3 = tf.keras.layers.Conv2D(64, 2, 1, name='conv3')
        self.prelu3 = tf.keras.layers.PReLU(shared_axes=[1,2], name="prelu3")

        self.dense4 = tf.keras.layers.Dense(128, name='conv4')
        self.prelu4 = tf.keras.layers.PReLU(shared_axes=None, name="prelu4")

        self.dense5_1 = tf.keras.layers.Dense(2, name="conv5-1")
        self.dense5_2 = tf.keras.layers.Dense(4, name="conv5-2")

    def call(self, x, training=False):
        out = self.prelu1(self.conv1(x))
        out = tf.nn.max_pool2d(out, 3, 2, padding="SAME")
        out = self.prelu2(self.conv2(out))
        out = tf.nn.max_pool2d(out, 3, 2, padding="VALID")
        out = self.prelu3(self.conv3(out))
        out = tf.reshape(out, shape=(out.shape[0], -1))
        out = self.prelu4(self.dense4(out))
        score = tf.nn.softmax(self.dense5_1(out), -1)
        boxes = self.dense5_2(out)
        return boxes, score

class ONet(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(32, 3, 1, name="conv1")
        self.prelu1 = tf.keras.layers.PReLU(shared_axes=[1,2], name="prelu1")

        self.conv2 = tf.keras.layers.Conv2D(64, 3, 1, name="conv2")
        self.prelu2 = tf.keras.layers.PReLU(shared_axes=[1,2], name="prelu2")

        self.conv3 = tf.keras.layers.Conv2D(64, 3, 1, name="conv3")
        self.prelu3 = tf.keras.layers.PReLU(shared_axes=[1,2], name="prelu3")

        self.conv4 = tf.keras.layers.Conv2D(128, 2, 1, name="conv4")
        self.prelu4 = tf.keras.layers.PReLU(shared_axes=[1,2], name="prelu4")

        self.dense5 = tf.keras.layers.Dense(256, name="conv5")
        self.prelu5 = tf.keras.layers.PReLU(shared_axes=None, name="prelu5")

        self.dense6_1 = tf.keras.layers.Dense(2  , name="conv6-1")
        self.dense6_2 = tf.keras.layers.Dense(4  , name="conv6-2")
        self.dense6_3 = tf.keras.layers.Dense(10 , name="conv6-3")

    def call(self, x, training=False):
        out = self.prelu1(self.conv1(x))
        out = tf.nn.max_pool2d(out, 3, 2, padding="SAME")
        out = self.prelu2(self.conv2(out))
        out = tf.nn.max_pool2d(out, 3, 2, padding="VALID")
        out = self.prelu3(self.conv3(out))
        out = tf.nn.max_pool2d(out, 2, 2, padding="SAME")
        out = self.prelu4(self.conv4(out))
        out = self.dense5(tf.reshape(out, shape=(out.shape[0], -1)))
        out = self.prelu5(out)
        score = tf.nn.softmax(self.dense6_1(out))
        boxes = self.dense6_2(out)
        lamks = self.dense6_3(out)
        return boxes, lamks, score

