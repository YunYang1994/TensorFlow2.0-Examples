#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-10-14 19:12:36
#   Description :
#
#================================================================

import tensorflow as tf
from fcn8s import FCN8s
from utils import DataGenerator

TrainSet = DataGenerator("./data/train_image.txt", "./data/train_labels", 2)
model = FCN8s(n_class=21)
callback = tf.keras.callbacks.ModelCheckpoint("FCN8s.h5", verbose=1, save_weights_only=True)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
              callback=callback,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

## train your FCN8s model
model.fit_generator(TrainSet, steps_per_epoch=6000, epochs=30)

