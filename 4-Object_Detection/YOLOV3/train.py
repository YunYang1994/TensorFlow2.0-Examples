#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-07-18 09:18:54
#   Description :
#
#================================================================

import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov3 import YOLOv3, decode, compute_loss
from core.config import cfg


trainset = Dataset('train')
steps_per_epoch    = len(trainset)

input_tensor = tf.keras.layers.Input([416, 416, 3])
conv_tensors = YOLOv3(input_tensor)

output_tensors = []
for i, conv_tensor in enumerate(conv_tensors):
    pred_tensor = decode(conv_tensor, i)
    output_tensors.append(conv_tensor)
    output_tensors.append(pred_tensor)

model = tf.keras.Model(input_tensor, output_tensors)

global_step = tf.Variable(1.0, dtype=tf.float32, trainable=False)
warmup_steps = tf.constant(cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch, dtype=tf.float32)

learn_rate = tf.cond(
    pred=global_step < warmup_steps,
    true_fn=lambda: global_step / warmup_steps * cfg.TRAIN.LEARN_RATE_INIT,
    false_fn=lambda: cfg.TRAIN.LEARN_RATE_END + 0.5 * (cfg.TRAIN.LEARN_RATE_INIT - cfg.TRAIN.LEARN_RATE_END) *
                        (1 + tf.cos((global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi)))

optimizer = tf.keras.optimizers.Adam(learn_rate)
def train_step(image_data, target):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        giou_loss=conf_loss=prob_loss=0

        for i in range(3):
            conv, pred = pred_result[i*2], pred_result[i*2+1]
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print(giou_loss, conf_loss, prob_loss, total_loss)


for epoch in range(cfg.TRAIN.EPOCHS):
    print("===========================> epoch %d, lr %.7f" %(epoch, learn_rate.numpy()))
    for image_data, target in trainset:
        train_step(image_data, target)
    global_step.assign_add(1.0)
    model.save_weights("./yolov3")

