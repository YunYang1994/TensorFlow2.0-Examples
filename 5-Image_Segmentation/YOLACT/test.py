#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : test.py
#   Author      : YunYang1994
#   Created date: 2019-07-29 20:24:48
#   Description :
#
#================================================================

import cv2
import torch
import numpy as np
import torch.nn as nn

image = cv2.imread("./boy.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (550, 550))
image = image / 255.
image = np.expand_dims(image, 0).astype(np.float32)

torch_image = np.transpose(image, [0, 3, 1, 2])
torch_image = torch.Tensor(torch_image)

tf_image = image

bn = nn.BatchNorm2d(32)
torch.nn.init.normal(bn.running_mean)
torch.nn.init.constant(bn.running_var, 100)

model = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
        bn,
        nn.LeakyReLU(0.1, inplace=True))

with torch.no_grad():
    model.eval()
    torch_result = model(torch_image)
    torch_result = np.transpose(torch_result, [0, 2, 3, 1])

# save_result
layers = model.children()
conv_layer = next(layers)
conv_weight = conv_layer.weight.detach().numpy()
conv_weight = np.transpose(conv_weight, [2, 3, 1, 0])
bn_layer = next(layers)

gama = bn_layer.weight.detach().numpy()
beta = bn_layer.bias.detach().numpy()
running_mean = bn_layer.running_mean.detach().numpy()
running_var = bn_layer.running_var.detach().numpy()

import tensorflow as tf

input_layer = tf.keras.layers.Input([None, None, 3])
conv = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(input_layer)
bn = tf.keras.layers.BatchNormalization()(conv)
y = tf.nn.leaky_relu(bn, alpha=0.1)
tf_model = tf.keras.Model(input_layer, y)

tf_model.layers[1].set_weights([conv_weight])
tf_model.layers[2].set_weights([gama, beta, running_mean, running_var])

tf_result = tf_model(tf_image)


print(torch_result[0][0][0])
print(tf_result[0][0][0])


