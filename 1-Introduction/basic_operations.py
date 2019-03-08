#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : basic_operations.py
#   Author      : YunYang1994
#   Created date: 2019-03-08 14:32:57
#   Description :
#
#================================================================

"""
Basic Operations example using TensorFlow library.
"""

import tensorflow as tf

###======================================= assign value  ===================================#

a = tf.ones([2,3])
print(a)

# a[0,0] = 10 => TypeError: 'tensorflow.python.framework.ops.EagerTensor' object does not support item assignment

a = tf.Variable(a)
a[0,0].assign(10)
b = a.read_value()
print(b)

###======================================= add, multiply, div. etc ===================================#

a = tf.constant(2)
b = tf.constant(3)

print("a + b :" , a.numpy() + b.numpy())
print("Addition with constants: ", a+b)
print("Addition with constants: ", tf.add(a, b))
print("a * b :" , a.numpy() * b.numpy())
print("Multiplication with constants: ", a*b)
print("Multiplication with constants: ", tf.multiply(a, b))


# ----------------
# More in details:
# Matrix Multiplication from TensorFlow official tutorial

# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2)
print("Multiplication with matrixes:", product)

# broadcast matrix in Multiplication

print("broadcast matrix in Multiplication:", matrix1 * matrix2)


###===================================== cast operations =====================================#

a = tf.convert_to_tensor(2.)
b = tf.cast(a, tf.int32)
print(a, b)

###===================================== shape operations ===================================#

a = tf.ones([2,3])
print(a.shape[0], a.shape[1]) # 2, 3
shape = tf.shape(a)           # a tensor
print(shape[0], shape[1])


