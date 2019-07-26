#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : yolact.py
#   Author      : YunYang1994
#   Created date: 2019-07-26 13:54:39
#   Description :
#
#================================================================

import tensorflow as tf
from typing import List


def conv_layer(in_channels, out_channels, kernel_size, strides=1, padding=0):
    x = tf.keras.layers.Input([None, None, in_channels])
    y = tf.keras.layers.ZeroPadding2D(padding)(x)
    y = tf.keras.layers.Conv2D(out_channels, kernel_size, strides=strides, use_bias=False,
                               kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                               bias_initializer=tf.constant_initializer(0.))(y)
    return tf.keras.Model(x, y)



class FPN(tf.keras.Model):
    """
    Implements a general version of the FPN introduced in
    https://arxiv.org/pdf/1612.03144.pdf

    Args:
        - in_channels (list): For each conv layer you supply in the forward pass,
                              how many features will it have?
    """
    __constants__ = ['interpolation_mode', 'num_downsample', 'use_conv_downsample',
                     'lat_layers', 'pred_layers', 'downsample_layers']
    def __init__(self, in_channels):
        super(FPN, self).__init__()

        self.lat_layers  = [conv_layer(x,   256, kernel_size=1)
                                                       for _ in reversed(in_channels)]
        # This is here for backwards compatability
        self.pred_layers = [conv_layer(256, 256, kernel_size=3, padding=1)
                                                       for _ in in_channels]
        self.downsample_layers = [conv_layer(256, 256, kernel_size=1, padding=1, strides=2)
                                                       for _ in range(2)]

    def call(self, convouts:List[tf.Tensor]):
        pass















