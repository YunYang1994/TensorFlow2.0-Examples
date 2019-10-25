#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : fpn.py
#   Author      : YunYang1994
#   Created date: 2019-10-25 17:37:35
#   Description :
#
#================================================================

import tensorflow as tf

"""
Implements a ResNet version of the FPN introduced in
https://arxiv.org/pdf/1612.03144.pdf
"""

class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, in_channels, out_channels, strides=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=strides,
                                            padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=1,
                                            padding="same", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

        """
        Adds a shortcut between input and residual block and merges them with "sum"
        """
        self.shortcut = lambda x: x
        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(self.expansion*out_channels, kernel_size=1,
                                           strides=strides, use_bias=False),
                    tf.keras.layers.BatchNormalization()]
                    )

    def call(self, x, training=False):
        # if training: print("=> training network ... ")
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        out += self.shortcut(x)
        return tf.nn.relu(out)

class FPN(tf.keras.Model):
    """ use ResNet as backbone
    """
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_channels = 64

        self.conv1 = tf.keras.layers.Conv2D(64, 7, 2, padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()

        # Bottom --> up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.top_layer = tf.keras.layers.Conv2D(256, 1, 1, padding="valid")

        # Smooth layers
        self.smooth1   = tf.keras.layers.Conv2D(256, 3, 1, padding="same")
        self.smooth2   = tf.keras.layers.Conv2D(256, 3, 1, padding="same")
        self.smooth3   = tf.keras.layers.Conv2D(256, 3, 1, padding="same")

        # Lateral layers
        self.lateral_layer1 = tf.keras.layers.Conv2D(256, 1, 1, padding="valid")
        self.lateral_layer2 = tf.keras.layers.Conv2D(256, 1, 1, padding="valid")
        self.lateral_layer3 = tf.keras.layers.Conv2D(256, 1, 1, padding="valid")

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return tf.keras.Sequential(layers)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
          """
        _, H, W, C = y.shape
        return tf.image.resize(x, size=(H, W), method="bilinear")

    def call(self, x, training=False):
        p1 = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        p1 = tf.nn.max_pool2d(p1, ksize=3, strides=2, padding="SAME")

        # Bottom --> up
        p2 = self.layer1(p1)
        p3 = self.layer1(p2)
        p4 = self.layer1(p3)
        p5 = self.layer1(p4)

        # Top-down
        d5 = self.top_layer(p5)
        d4 = self._upsample_add(d5, self.lateral_layer1(p4))
        d3 = self._upsample_add(d4, self.lateral_layer2(p3))
        d2 = self._upsample_add(d3, self.lateral_layer3(p2))

        # Smooth
        d4 = self.smooth1(d4)
        d3 = self.smooth2(d3)
        d2 = self.smooth3(d2)

        return d2, d3, d4, d5

def ResNet18_fpn():
    return FPN(BasicBlock, [2, 2, 2, 2])

def ResNet34_fpn():
    return FPN(BasicBlock, [3, 4, 6, 3])

if __name__ == "__main__":
    ## Test model
    data = tf.ones(shape=[1, 416, 416, 3])
    # model = ResNet18_fpn()
    model = ResNet34_fpn()
    fms = model(data)
    for fm in fms:
        print(fm.shape)


