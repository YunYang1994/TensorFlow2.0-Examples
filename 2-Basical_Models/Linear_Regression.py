#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : Linear_Regression.py
#   Author      : YunYang1994
#   Created date: 2019-03-08 17:33:48
#   Description :
#
#================================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define model and Loss

class Model(object):
    def __init__(self):
        self.W = tf.Variable(10.0)
        self.b = tf.Variable(-5.0)

    def __call__(self, inputs):
        return self.W * inputs + self.b

def compute_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred))

model = Model()

# Define True weight and bias

TRUE_W = 3.0
TRUE_b = 2.0

# Obtain training data, Let's synthesize the training data with some noise.

NUM_EXAMPLES = 1000
inputs  = tf.random.normal(shape=[NUM_EXAMPLES])
noise   = tf.random.normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

# Before we train the model let's visualize where the model stands right now.
# We'll plot the model's predictions in red and the training data in blue.

def plot(epoch):
    plt.scatter(inputs, outputs, c='b')
    plt.scatter(inputs, model(inputs), c='r')
    plt.title("epoch %2d, loss = %s" %(epoch, str(compute_loss(outputs, model(inputs)).numpy())))
    plt.legend()
    plt.draw()
    plt.ion()   # replacing plt.show()
    plt.pause(1)
    plt.close()

# Define a training loop
learning_rate = 0.1
for epoch in range(30):
    with tf.GradientTape() as tape:
        loss = compute_loss(outputs, model(inputs))

    dW, db = tape.gradient(loss, [model.W, model.b])

    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)

    print("=> epoch %2d: w_true= %.2f, w_pred= %.2f; b_true= %.2f, b_pred= %.2f, loss= %.2f" %(
          epoch+1, TRUE_W, model.W.numpy(), TRUE_b, model.b.numpy(), loss.numpy()))
    plot(epoch + 1)

