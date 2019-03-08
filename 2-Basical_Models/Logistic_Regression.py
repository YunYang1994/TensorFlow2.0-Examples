#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : Logistic_Regression.py
#   Author      : YunYang1994
#   Created date: 2019-03-08 22:28:21
#   Description :
#
#================================================================

import numpy as np
import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 6
batch_size = 600

# Import MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

train_dataset = (
    tf.data.Dataset.from_tensor_slices((tf.reshape(x_train, [-1, 784]), y_train))
    .batch(batch_size)
    .shuffle(1000)
)

train_dataset = (
    train_dataset.map(lambda x, y:
                      (tf.divide(tf.cast(x, tf.float32), 255.0),
                       tf.reshape(tf.one_hot(y, 10), (-1, 10))))
)


# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
model = lambda x: tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
# Minimize error using cross entropy
compute_loss = lambda true, pred: tf.reduce_mean(tf.reduce_sum(tf.losses.binary_crossentropy(true, pred), axis=-1))
# caculate accuracy
compute_accuracy = lambda true, pred: tf.reduce_mean(tf.keras.metrics.categorical_accuracy(true, pred))
# Gradient Descent
optimizer = tf.optimizers.Adam(learning_rate)

for epoch in range(training_epochs):
    for i, (x_, y_) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            pred = model(x_)
            loss = compute_loss(y_, pred)
        acc = compute_accuracy(y_, pred)
        grads = tape.gradient(loss, [W, b])
        optimizer.apply_gradients(zip(grads, [W, b]))
        print("=> loss %.2f acc %.2f" %(loss.numpy(), acc.numpy()))

