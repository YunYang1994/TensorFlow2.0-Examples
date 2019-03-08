#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : GradientTape.py
#   Author      : YunYang1994
#   Created date: 2019-03-08 13:50:49
#   Description :
#
#================================================================

import tensorflow as tf

# tf.GradientTape is an API for automatic differentiation - computing the gradient of
# a computation with respect to its input variables. Tensorflow "records" all operations
# executed inside the context of a tf.GradientTape onto a "tape"

## Automatic differentiation

x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:
    t.watch(x)       # Ensures that `tensor` is being traced by this tape.
    y = x * x
    z = y * y
dz_dx = t.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
dy_dx = t.gradient(y, x)  # 6.0
print("dz/dx=", dz_dx.numpy())
print("dy/dx=", dy_dx.numpy())
del t  # Drop the reference to the tape


## Recording control flow

# Because tapes record operations as they are executed,
# Python control flow (using ifs and whiles for example) is naturally handled
def f(x, y):
    output = 1.0
    for i in range(y):
        if i > 1 and i < 5:
            output = tf.multiply(output, x)
    return output

def grad(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        out = f(x, y)
    return t.gradient(out, x)

x = tf.convert_to_tensor(2.0)

assert grad(x, 6).numpy() == 12.0
assert grad(x, 5).numpy() == 12.0
assert grad(x, 4).numpy() == 4.0


