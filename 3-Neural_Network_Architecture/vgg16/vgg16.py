#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : vgg16.py
#   Author      : YunYang1994
#   Created date: 2019-07-16 10:09:17
#   Description :
#
#================================================================

import skimage
import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]


# define input layer
input_layer = tf.keras.layers.Input([224, 224, 3])

red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=input_layer)
bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]])

# Block 1
conv1_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                 use_bias=True, activation='relu', name='conv1_1')(bgr)

conv1_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                 use_bias=True, activation='relu', name='conv1_2')(conv1_1)
pool1_1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1_1')

# Block 2
conv2_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                 use_bias=True, activation='relu', name='conv2_1')(pool1_1)
conv2_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                 use_bias=True, activation='relu', name='conv2_2')(conv2_1)
pool2_1 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2_1')

# Block 3
conv3_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                 use_bias=True, activation='relu', name='conv3_1')(pool2_1)
conv3_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                 use_bias=True, activation='relu', name='conv3_2')(conv3_1)
conv3_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                 use_bias=True, activation='relu', name='conv3_3')(conv3_2)
pool3_1 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3_1')

# Block 4
conv4_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                 use_bias=True, activation='relu', name='conv4_1')(pool3_1)
conv4_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                 use_bias=True, activation='relu', name='conv4_2')(conv4_1)
conv4_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                 use_bias=True, activation='relu', name='conv4_3')(conv4_2)
pool4_1 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4_1')

# Block 4
conv5_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                 use_bias=True, activation='relu', name='conv5_1')(pool4_1)
conv5_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                 use_bias=True, activation='relu', name='conv5_2')(conv5_1)
conv5_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                 use_bias=True, activation='relu', name='conv5_3')(conv5_2)
pool5_1 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5_1')

flatten = tf.keras.layers.Flatten()(pool5_1)
fc6 = tf.keras.layers.Dense(units=4096, use_bias=True, name='fc6', activation='relu')(flatten)
fc7 = tf.keras.layers.Dense(units=4096, use_bias=True, name='fc7', activation='relu')(fc6)
fc8 = tf.keras.layers.Dense(units=1000, use_bias=True, name='fc8', activation=None)(fc7)

prob = tf.nn.softmax(fc8)

# Build model
model = tf.keras.Model(input_layer, prob)

# Load weighs
weighs = np.load("./vgg16.npy", encoding='latin1').item()
for layer_name in weighs.keys():
    layer = model.get_layer(layer_name)
    layer.set_weights(weighs[layer_name])

# Load image
image_data = skimage.io.imread("./docs/cat.jpg").astype(np.float32)

# Load labels
labels = open("./docs/synset_words.txt", "r").readlines()

# Print result
print(labels[np.argmax(model(np.expand_dims(image_data, 0)))])





