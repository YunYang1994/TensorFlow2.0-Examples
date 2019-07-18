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
from core.yolov3 import YOLOV3
from core.config import cfg

class YoloTrain(object):
    def __init__(self):
        self.anchor_per_scale    = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes             = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes         = len(self.classes)
        self.learn_rate_init     = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end      = cfg.TRAIN.LEARN_RATE_END
        self.epochs              = cfg.TRAIN.EPOCHS
        self.warmup_periods      = cfg.TRAIN.WARMUP_EPOCHS
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.max_bbox_per_scale  = 150
        self.train_logdir        = "./data/log/train"
        self.trainset            = Dataset('train')
        self.steps_per_period    = len(self.trainset)
        self.input_layer         = tf.keras.layers.Input([416, 416, 3])
        self.model               = YOLOV3(self.input_layer, training=True)
        self.trainable_variables = self.model.model.trainable_variables

        # self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
        # warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period, dtype=tf.float64, name='warmup_steps')
        # train_steps = tf.constant( self.epochs * self.steps_per_period, dtype=tf.float64, name='train_steps')
        # self.learn_rate = tf.cond(
            # pred=self.global_step < warmup_steps,
            # true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
            # false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                # (1 + tf.cos(
                                    # (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
        # )
        # global_step_update = tf.assign_add(self.global_step, 1.0)
        self.optimizer = tf.keras.optimizers.Adam(0.0001)

    # @tf.function
    def train_step(self, input_image, target):
        with tf.GradientTape() as tape:
            giou_loss, conf_loss, prob_loss = self.model.compute_loss(input_image, target)
            total_loss = giou_loss + conf_loss + prob_loss
            gradients = tape.gradient(total_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return giou_loss, conf_loss, prob_loss, total_loss



    def train(self):

        pbar = tqdm(self.trainset)
        train_epoch_loss, test_epoch_loss = [], []

        for train_data in pbar:
            input_image, target = train_data[0], train_data[1:]
            input_image = tf.convert_to_tensor(input_image)
            target = (tf.convert_to_tensor(x) for x in target)
            giou_loss, conf_loss, prob_loss, total_loss = self.train_step(input_image, target)
            self.model.model.save_weights("yolov3")
            print(total_loss)



if __name__ == '__main__': YoloTrain().train()
