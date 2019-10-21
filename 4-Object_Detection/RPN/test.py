#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : test.py
#   Author      : YunYang1994
#   Created date: 2019-10-19 17:21:54
#   Description :
#
#================================================================

import numpy as np
import tensorflow as tf
from utils import wandhG

wandhG = np.array(wandhG)


grid_h = 45
grid_w = 60

grid_x, grid_y = tf.range(grid_w, dtype=tf.int32), tf.range(grid_h, dtype=tf.int32)
grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
grid_x, grid_y = tf.expand_dims(grid_x, -1), tf.expand_dims(grid_y, -1)
grid_xy = tf.stack([grid_x, grid_y], axis=-1)

