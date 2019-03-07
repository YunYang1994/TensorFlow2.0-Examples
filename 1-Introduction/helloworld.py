#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : helloworld.py
#   Author      : YunYang1994
#   Created date: 2019-03-08 00:21:22
#   Description :
#
#================================================================

import tensorflow as tf

# Simple hello world using TensorFlow

# Create a Constant op
# The op is added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.

helloworld = tf.constant("hello, TensorFlow")
print("Tensor:", helloworld)
print("Value :", helloworld.numpy())

