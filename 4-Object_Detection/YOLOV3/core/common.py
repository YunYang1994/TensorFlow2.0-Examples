#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : common.py
#   Author      : YunYang1994
#   Created date: 2019-07-11 23:12:53
#   Description :
#
#================================================================

import tensorflow as tf


def convolutional(input_data, filters_shape, downsample=False, activate=True, bn=True):
    if downsample:
        input_data = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_data)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(input_data)

    if bn:
        conv = tf.keras.layers.BatchNormalization()(conv)

    if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv

def residual_block(input_data, input_channel, filter_num1, filter_num2):
    short_cut = input_data
    conv = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1))
    conv = convolutional(conv      , filters_shape=(3, 3, filter_num1,   filter_num2))

    residual_output = short_cut + conv
    return residual_output

def upsample(input_data):
    return tf.image.resize(input_data, (input_data.shape[1] * 2, input_data.shape[2] * 2), method='nearest')

