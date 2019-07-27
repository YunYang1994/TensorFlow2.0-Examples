#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : backbone.py
#   Author      : YunYang1994
#   Created date: 2019-07-25 17:43:16
#   Description :
#
#================================================================

import tensorflow as tf

def darknetconvlayer(in_channels, out_channels, kernel_size, strides=1, padding=0):
    """
    Implements a conv, activation, then batch norm.
    Arguments are passed into the conv layer.
    """
    x = tf.keras.layers.Input([None, None, in_channels])
    y = tf.keras.layers.ZeroPadding2D(padding)(x)
    y = tf.keras.layers.Conv2D(out_channels, kernel_size, strides=strides, use_bias=False,
    						   kernel_initializer=tf.random_normal_initializer(stddev=0.01),
    						   bias_initializer=tf.constant_initializer(0.))(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.nn.leaky_relu(y, alpha=0.1)

    return tf.keras.Model(x, y)


class DarkNetBlock(tf.keras.Model):
    """ Note: channels is the lesser of the two. The output will be expansion * channels. """

    expansion = 2

    def __init__(self, in_channels, out_channels):
        super(DarkNetBlock, self).__init__()

        self.conv1 = darknetconvlayer(in_channels,  out_channels,                  kernel_size=1)
        self.conv2 = darknetconvlayer(out_channels, out_channels * self.expansion, kernel_size=3, padding=1)

    def call(self, x):
        return self.conv2(self.conv1(x)) + x


class DarkNetBackbone(tf.keras.Model):
    """
    An implementation of YOLOv3's Darnet53 in
    https://pjreddie.com/media/files/papers/YOLOv3.pdf

    This is based off of the implementation of Resnet above.
    """

    def __init__(self, layers=[1, 2, 8, 8, 4], block=DarkNetBlock):
        super(DarkNetBackbone, self).__init__()

        # These will be populated by _make_layer
        self.num_base_layers = len(layers)
        self.conv_layers = []
        self.channels = []

        self._preconv = darknetconvlayer(3, 32, kernel_size=3, padding=1)
        self.in_channels = 32

        self._make_layer(block, 32,  layers[0])
        self._make_layer(block, 64,  layers[1])
        self._make_layer(block, 128, layers[2])
        self._make_layer(block, 256, layers[3])
        self._make_layer(block, 512, layers[4])

    def _make_layer(self, block, channels, num_blocks, strides=2):
        """ Here one layer means a string of n blocks. """

        layer_list = []
        # The downsample layer
        layer_list.append(
            darknetconvlayer(self.in_channels, channels * block.expansion,
                             kernel_size=3, strides=strides, padding=1))

        # Each block inputs channels and outputs channels * expansion
        self.in_channels = channels * block.expansion
        for _ in range(num_blocks):
        	layer_list.append(block(self.in_channels, channels))

        self.channels.append(self.in_channels)
        self.conv_layers.append(tf.keras.Sequential(layer_list))

    def call(self, x):
        """ Returns a list of convouts for each layer. """

        y = self._preconv(x)

        outs = []
        for layer in self.conv_layers:
            y = layer(y)
            outs.append(y)

        return tuple(outs)

    def add_layer(self, conv_channels=1024, strides=2, depth=1, block=DarkNetBlock):
        """ Add a downsample layer to the backbone as per what SSD does. """
        self._make_layer(block, conv_channels // block.expansion, num_blocks=depth, strides=strides)


darknet53 = DarkNetBackbone()






