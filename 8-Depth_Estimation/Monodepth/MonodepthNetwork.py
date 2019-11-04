#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : MonodepthNetwork.py
#   Author      : YunYang1994
#   Created date: 2019-11-01 23:57:18
#   Description :
#
#================================================================

import tensorflow as tf
from utils import upsample_nn

class Bottleneck(tf.keras.Model):
    expansion = 4

    def __init__(self, in_channels, out_channels, strides=1):
        super(Bottleneck, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(out_channels, 1, 1, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(out_channels, 3, strides, padding="same", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(out_channels*self.expansion, 1, 1, use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()

        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(self.expansion*out_channels, kernel_size=1,
                                           strides=strides, use_bias=False),
                    tf.keras.layers.BatchNormalization()]
                    )
        else:
            self.shortcut = lambda x,_: x

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = tf.nn.relu(self.bn2(self.conv2(out), training=training))
        out = self.bn3(self.conv3(out), training=training)
        out += self.shortcut(x, training)
        return tf.nn.relu(out)

class resnet_encoder(tf.keras.Model):
    """ take resnet50 as backbone"""
    def __init__(self):
        super(resnet_encoder, self).__init__()

        self.in_channels = 64
        # Encode layers
        self.conv1 = tf.keras.layers.Conv2D(64, 7, 2, padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pad = tf.keras.layers.ZeroPadding2D(padding=1)
        self.pool1 = tf.keras.layers.MaxPool2D(3, 2, padding="valid") # H/4  -   64D
        self.layer1 = self._make_layer(Bottleneck,  64, 3, 2)         # H/8  -  256D
        self.layer2 = self._make_layer(Bottleneck, 128, 4, 2)         # H/16 -  512D
        self.layer3 = self._make_layer(Bottleneck, 256, 6, 2)         # H/32 - 1024D
        self.layer4 = self._make_layer(Bottleneck, 512, 3, 2)         # H/64 - 2048D

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return tf.keras.Sequential(layers)

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training))
        skip1 = out
        out = self.pool1(self.pad(out))
        skip2 = out
        out = self.layer1(out, training=training)
        skip3 = out
        out = self.layer2(out, training=training)
        skip4 = out
        out = self.layer3(out, training=training)
        skip5 = out
        out = self.layer4(out, training=training)
        return skip1, skip2, skip3, skip4, skip5, out


class upsample_conv(tf.keras.Model):
    def __init__(self, filters, kernel_size, scale):
        super(upsample_conv, self).__init__()
        self.scale = scale
        self.conv = tf.keras.Sequential([tf.keras.layers.ZeroPadding2D(1),
                tf.keras.layers.Conv2D(filters, kernel_size, use_bias=False,
                                       strides=1, padding="valid", activation="relu")]
                )

    def call(self, x, training=False):
        out = upsample_nn(x, self.scale)
        out = self.conv(out)
        return out

class depth_decoder(tf.keras.Model):
    def __init__(self):
        super(depth_decoder, self).__init__()
        self.out_channels = [16, 32, 64, 128, 256, 512]
        self.upsample_conv_layers = [upsample_conv(x, 3, 2) for x in self.out_channels]
        self.conv_layers = [self._conv3x3(x, 3, 1, "relu") for x in self.out_channels]
        self.disp_layers = [self._conv3x3(2, 3, 1, "sigmoid") for _ in range(4)]

    def _conv3x3(self, out_channels, kernel_size, strides, activation_fn):
        return tf.keras.Sequential([
                tf.keras.layers.ZeroPadding2D(1),
                tf.keras.layers.Conv2D(out_channels, kernel_size, strides,
                    padding="valid", use_bias=False, activation=activation_fn)]
                )


    def call(self, input_features, training=False):
        skip1, skip2, skip3, skip4, skip5, out = input_features

        upconv5 = self.upsample_conv_layers[5](out) # H/32 - 512D
        concat5 = tf.concat([upconv5, skip5], axis=3)
        iconv5  = self.conv_layers[5](concat5)

        upconv4 = self.upsample_conv_layers[4](iconv5) # H/16 - 256D
        concat4 = tf.concat([upconv4, skip4], axis=3)
        iconv4  = self.conv_layers[4](concat4)

        upconv3 = self.upsample_conv_layers[3](iconv4)
        concat3 = tf.concat([upconv3, skip3], axis=3)
        iconv3  = self.conv_layers[3](concat3)
        disp3 = 0.3 * self.disp_layers[3](iconv3) # H/8 - 2D
        # print(disp3.shape)
        up_disp3 = upsample_nn(disp3, 2)

        upconv2 = self.upsample_conv_layers[2](iconv3)
        concat2 = tf.concat([upconv2, skip2, up_disp3], axis=3)
        iconv2  = self.conv_layers[2](concat2)
        disp2 = 0.3 * self.disp_layers[2](iconv2) # H/4 - 2D
        # print(disp2.shape)
        up_disp2 = upsample_nn(disp2, 2)

        upconv1 = self.upsample_conv_layers[1](iconv2)
        concat1 = tf.concat([upconv1, skip1, up_disp2], axis=3)
        iconv1  = self.conv_layers[1](concat1)
        disp1 = 0.3 * self.disp_layers[1](iconv1) # H/2 - 2D
        # print(disp1.shape)
        up_disp1 = upsample_nn(disp1, 2)

        upconv0 = self.upsample_conv_layers[0](iconv1)
        concat0 = tf.concat([upconv0, up_disp1], axis=3)
        iconv0  = self.conv_layers[0](concat0)
        disp0 = 0.3 * self.disp_layers[0](iconv0) # H - 2D
        # print(disp0.shape)
        return disp0, disp1, disp2, disp3

class MonodepthNetwork(tf.keras.Model):
    def __init__(self):
        super(MonodepthNetwork, self).__init__()
        self.encoder = resnet_encoder()
        self.decoder = depth_decoder()

    def call(self, x, training=False):
        out = self.encoder(x, training=training)
        out = self.decoder(out, training=training)
        return out


