#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-10-17 15:00:25
#   Description :
#
#================================================================

import tensorflow as tf
from rpn import RPNplus

data = tf.ones(shape=[1, 720, 960, 3], dtype=tf.float32)
model = RPNplus()
y = model(data)
