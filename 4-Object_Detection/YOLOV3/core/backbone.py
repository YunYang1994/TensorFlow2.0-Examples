#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : backbone.py
#   Author      : YunYang1994
#   Created date: 2019-07-11 23:37:51
#   Description :
#
#================================================================

import tensorflow as tf
import core.common as common


def darknet53(input_data, training=False):

    input_data = common.convolutional(input_data, (3, 3,  3,  32), training)
    input_data = common.convolutional(input_data, (3, 3, 32,  64),  training, downsample=True)

    for i in range(1):
        input_data = common.residual_block(input_data,  64,  32, 64, training)

    input_data = common.convolutional(input_data, (3, 3,  64, 128), training, downsample=True)

    for i in range(2):
        input_data = common.residual_block(input_data, 128,  64, 128, training)

    input_data = common.convolutional(input_data, (3, 3, 128, 256), training, downsample=True)

    for i in range(8):
        input_data = common.residual_block(input_data, 256, 128, 256, training)

    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 256, 512), training, downsample=True)

    for i in range(8):
        input_data = common.residual_block(input_data, 512, 256, 512, training)

    route_2 = input_data
    input_data = common.convolutional(input_data, (3, 3, 512, 1024), training, downsample=True)

    for i in range(4):
        input_data = common.residual_block(input_data, 1024, 512, 1024, training)

    return route_1, route_2, input_data


