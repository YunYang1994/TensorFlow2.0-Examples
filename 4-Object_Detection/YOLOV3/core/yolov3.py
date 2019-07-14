#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : yolov3.py
#   Author      : YunYang1994
#   Created date: 2019-07-12 13:47:10
#   Description :
#
#================================================================

import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg

class YOLOV3(object):
     """Implement tensoflow yolov3 here"""
     def __init__(self, input_layer):

        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_class        = len(self.classes)
        self.strides          = np.array(cfg.YOLO.STRIDES)
        self.anchors          = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh  = cfg.YOLO.IOU_LOSS_THRESH
        self.upsample_method  = cfg.YOLO.UPSAMPLE_METHOD

        try:
            self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self.__build_nework(input_layer)
        except:
            raise NotImplementedError("Can not build up yolov3 network!")

        self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0])
        self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])
        self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])

        self.model = tf.keras.Model(input_layer, [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox])
        self.model.summary()

     def load_weights(self, weights_file):
        """
        I agree that this code is very ugly, but I don’t know any better way of doing it.
        """
        wf = open(weights_file, 'rb')
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

        j = 0
        for i in range(75):
            conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
            bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

            conv_layer = self.model.get_layer(conv_layer_name)
            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]

            if i not in [58, 66, 74]:
                # darknet weights: [beta, gamma, mean, variance]
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                # tf weights: [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                bn_layer = self.model.get_layer(bn_layer_name)
                j += 1
            else:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if i not in [58, 66, 74]:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

        assert len(wf.read()) == 0, 'failed to read all data'
        wf.close()

     def decode(self, conv_output, anchors, stride):
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """

        conv_shape       = tf.shape(conv_output)
        batch_size       = conv_shape[0]
        output_size      = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5: ]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

     def inference(self, image_data):
        return self.model(image_data)

     def __build_nework(self, input_layer):

        route_1, route_2, conv = backbone.darknet53(input_layer)

        conv = common.convolutional(conv, (1, 1, 1024,  512))
        conv = common.convolutional(conv, (3, 3,  512, 1024))
        conv = common.convolutional(conv, (1, 1, 1024,  512))
        conv = common.convolutional(conv, (3, 3,  512, 1024))
        conv = common.convolutional(conv, (1, 1, 1024,  512))

        conv_lobj_branch = common.convolutional(conv, (3, 3, 512, 1024))
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3*(80 + 5)), activate=False, bn=False)

        conv = common.convolutional(conv, (1, 1,  512,  256))
        conv = common.upsample(conv)

        conv = tf.concat([conv, route_2], axis=-1)

        conv = common.convolutional(conv, (1, 1, 768, 256))
        conv = common.convolutional(conv, (3, 3, 256, 512))
        conv = common.convolutional(conv, (1, 1, 512, 256))
        conv = common.convolutional(conv, (3, 3, 256, 512))
        conv = common.convolutional(conv, (1, 1, 512, 256))

        conv_mobj_branch = common.convolutional(conv, (3, 3, 256, 512))
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3*(80 + 5)), activate=False, bn=False)

        conv = common.convolutional(conv, (1, 1, 256, 128))
        conv = common.upsample(conv)

        conv = tf.concat([conv, route_1], axis=-1)

        conv = common.convolutional(conv, (1, 1, 384, 128))
        conv = common.convolutional(conv, (3, 3, 128, 256))
        conv = common.convolutional(conv, (1, 1, 256, 128))
        conv = common.convolutional(conv, (3, 3, 128, 256))
        conv = common.convolutional(conv, (1, 1, 256, 128))

        conv_sobj_branch = common.convolutional(conv, (3, 3, 128, 256))
        conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3*(80 +5)), activate=False, bn=False)

        return conv_lbbox, conv_mbbox, conv_sbbox

