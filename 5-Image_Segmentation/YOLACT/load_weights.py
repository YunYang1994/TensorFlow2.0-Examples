#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : utils.py
#   Author      : YunYang1994
#   Created date: 2019-07-28 16:09:16
#   Description :
#
#================================================================

import cv2
import numpy as np
import tensorflow as tf
from backbone import darknet53
from yolact import Protonet, FPN, Yolact, PredictionModule

weights_dict = np.load("yolact_darknet53_54_800000.npy", encoding='latin1').item()
darknet53_weights = list(reversed(weights_dict["darknet53"]))
proto_net_weights = list(reversed(weights_dict["proto_net"]))
fpn_weights = list(reversed(weights_dict["fpn"]))
pred_weights = list(reversed(weights_dict["pred"]))
segmantic_seg_conv_weights = list(reversed(weights_dict["segmantic_seg_conv"]))

def parse_module(module, weights):
    assert isinstance(module, tf.keras.Model)
    children = module.layers

    for child in children:
        if isinstance(child, tf.keras.Model):
            parse_module(child, weights)
        elif isinstance(child, tf.keras.layers.Conv2D):
            layer_weights = weights.pop()
            print(child, layer_weights[0].shape)
            child.set_weights(layer_weights)
        elif isinstance(child, tf.keras.layers.BatchNormalization):
            print(child, layer_weights[0].shape)
            layer_weights = weights.pop()
            child.set_weights(layer_weights)
        else:
            continue
    return True

model = Yolact()
darknet53_modules = [model.backbone._preconv] + model.backbone.conv_layers
for module in darknet53_modules:
    parse_module(module, darknet53_weights)

proto_net = model.proto_net
parse_module(proto_net, proto_net_weights)

fpn = model.fpn
parse_module(fpn, fpn_weights)

pred = model.prediction_layers[0]
parse_module(pred, pred_weights)

segmantic_seg_conv = model.semantic_seg_conv
parse_module(segmantic_seg_conv, segmantic_seg_conv_weights)


image = cv2.imread("./boy.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (550, 550))
image = image / 255.
image = np.expand_dims(image, 0).astype(np.float32)
result = model(image)





