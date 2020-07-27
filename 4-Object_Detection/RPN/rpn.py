#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : rpn.py
#   Author      : YunYang1994
#   Created date: 2019-10-17 13:46:28
#   Description :
#
#================================================================

import tensorflow as tf

class RPNplus(tf.keras.Model):
    # VGG_MEAN = [103.939, 116.779, 123.68]
    def __init__(self):
        super(RPNplus, self).__init__()
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
        self.pool5   = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')

        ## region_proposal_conv
        self.region_proposal_conv1 = tf.keras.layers.Conv2D(256, kernel_size=[5,2],
                                                            activation=tf.nn.relu,
                                                            padding='same', use_bias=False)
        self.region_proposal_conv2 = tf.keras.layers.Conv2D(512, kernel_size=[5,2],
                                                            activation=tf.nn.relu,
                                                            padding='same', use_bias=False)
        self.region_proposal_conv3 = tf.keras.layers.Conv2D(512, kernel_size=[5,2],
                                                            activation=tf.nn.relu,
                                                            padding='same', use_bias=False)
        ## Bounding Boxes Regression layer
        self.bboxes_conv = tf.keras.layers.Conv2D(36, kernel_size=[1,1],
                                                padding='same', use_bias=False)
        ## Output Scores layer
        self.scores_conv = tf.keras.layers.Conv2D(18, kernel_size=[1,1],
                                                padding='same', use_bias=False)


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
        # Pooling to same size
        pool3_p = tf.nn.max_pool2d(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool3_proposal')
        pool3_p = self.region_proposal_conv1(pool3_p) # [1, 45, 60, 256]

        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv4_3(h)
        h = self.pool4(h)
        pool4_p = self.region_proposal_conv2(h) # [1, 45, 60, 512]

        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = self.conv5_3(h)
        pool5_p = self.region_proposal_conv3(h) # [1, 45, 60, 512]

        region_proposal = tf.concat([pool3_p, pool4_p, pool5_p], axis=-1) # [1, 45, 60, 1280]

        conv_cls_scores = self.scores_conv(region_proposal) # [1, 45, 60, 18]
        conv_cls_bboxes = self.bboxes_conv(region_proposal) # [1, 45, 60, 36]

        cls_scores = tf.reshape(conv_cls_scores, [-1, 45, 60, 9, 2])
        cls_bboxes = tf.reshape(conv_cls_bboxes, [-1, 45, 60, 9, 4])

        return cls_scores, cls_bboxes

