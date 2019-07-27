#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : yolact.py
#   Author      : YunYang1994
#   Created date: 2019-07-26 13:54:39
#   Description :
#
#================================================================

import tensorflow as tf
from backbone import darknet53
from typing import List


def conv_layer(in_channels, out_channels, kernel_size, strides=1, padding=0, activation=None):
    x = tf.keras.layers.Input([None, None, in_channels])
    y = tf.keras.layers.ZeroPadding2D(padding)(x)
    y = tf.keras.layers.Conv2D(out_channels, kernel_size, strides, use_bias=False, activation=activation,
                               kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                               bias_initializer=tf.constant_initializer(0.))(y)
    return tf.keras.Model(x, y)

class InterpolateModule(tf.keras.Model):
    """Upsample Layer"""
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def call(self, x):
        _, h, w, _ = x.shape
        new_h = self.scale_factor * h
        new_w = self.scale_factor * w
        return tf.image.resize(x, size=(new_h, new_w))

class Protonet(tf.keras.Model):
    """
    Implements Protonet Architecture
    """
    def __init__(self):
        super(Protonet, self).__init__()

        self.conv_layers = [conv_layer(256, 256, 3, padding=1) for _ in range(3)]

        self.conv_layers.append(InterpolateModule(2))
        self.conv_layers.append(conv_layer(256, 256, 3, padding=1))
        self.conv_layers.append(conv_layer(256, 32, 1))

    def call(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        return x







class FPN(tf.keras.Model):
    """
    Implements a general version of the FPN introduced in
    https://arxiv.org/pdf/1612.03144.pdf

    Args:
        - in_channels (list): For each conv layer you supply in the forward pass,
                              how many features will it have?
    """
    __constants__ = ['interpolation_mode', 'num_downsample', 'use_conv_downsample',
                     'lat_layers', 'pred_layers', 'downsample_layers']
    def __init__(self, in_channels):
        super(FPN, self).__init__()

        self.lat_layers  = [conv_layer(x,   256, kernel_size=1)
                                                       for _ in reversed(in_channels)]
        # This is here for backwards compatability
        self.pred_layers = [conv_layer(256, 256, kernel_size=3, padding=1)
                                                       for _ in in_channels]
        self.downsample_layers = [conv_layer(256, 256, kernel_size=1, padding=1, strides=2)
                                                       for _ in range(2)]

    def call(self, convouts:List[tf.Tensor]):
        """
        Args:
            - convouts (list): A list of convouts for the corresponding layers in in_channels.
        Returns:
            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
        """
        out = []
        x = tf.zeros(1)
        for i in range(len(convouts)):
            out.append(x)

        # For backward compatability, the conv layers are stored in reverse but the input and output is
        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
        j = len(convouts)
        for lat_layer in self.lat_layers:
            j -= 1

            if j < len(convouts) - 1:
                _, h, w, _ = convouts[j].shape
                x = tf.image.resize(x, size=(h, w), method="bilinear")

            x = x + lat_layer(convouts[j])
            out[j] = x

        j = len(convouts)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = tf.nn.relu(pred_layer(out[j]))


        for downsample_layer in self.downsample_layers:
            out.append(downsample_layer(out[-1]))

        return out


class PredictionModule(tf.keras.Model):
    """
    The (c) prediction module adapted from DSSD:
    https://arxiv.org/pdf/1701.06659.pdf

    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution. Though, I really can't
    be arsed to implement it myself, and, who knows, this might be
    better.

    Args:
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
        - parent:        If parent is a PredictionModule, this module will use all the layers
                         from parent instead of from this module.
    """

    def __init__(self, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1], parent=None):
        super().__init__()

        self.num_classes = 81
        self.mask_dim    = 32
        self.num_priors  = sum(len(x) for x in aspect_ratios)
        self.parent      = [parent] # Don't include this in the state dict

        if parent is None:
            self.upfeature  = conv_layer(in_channels, 256, 3, padding=1, activation='relu')
            self.bbox_layer = conv_layer(3, self.num_priors * 4,                3, padding=1)
            self.conf_layer = conv_layer(3, self.num_priors * self.num_classes, 3, padding=1)
            self.mask_layer = conv_layer(3, self.num_priors * self.mask_dim,    3, padding=1)

            self.bbox_extra, self.conf_extra, self.mask_extra = lambda x:x, lambda x:x, lambda x:x

        self.aspect_ratios = aspect_ratios
        self.scales = scales

        self.priors = None
        self.last_conv_size = None

    def call(self, x):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])

        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        src = self if self.parent[0] is None else self.parent[0]

        conv_h, conv_w = x.shape[1:2]

        x = src.upfeature(x)
        bbox_x = src.bbox_extra(x)
        conf_x = src.conf_extra(x)
        mask_x = src.mask_extra(x)

        bbox = tf.reshape(src.bbox_layer(bbox_x), [-1, 4])
        conf = tf.reshape(src.conf_layer(conf_x), [-1, self.num_classes])
        mask = tf.reshape(src.mask_layer(mask_x), [-1, self.mask_dim])

        mask = torch.tanh(mask)
        priors = self.make_priors(conv_h, conv_w)

        return { 'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors }

    def make_priors(self, conv_h, conv_w):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """

        if self.last_conv_size != (conv_w, conv_h):
            prior_data = []

            # Iteration order is important (it has to sync up with the convout)
            for j, i in product(range(conv_h), range(conv_w)):
                # +0.5 because priors are in center-size notation
                x = (i + 0.5) / conv_w
                y = (j + 0.5) / conv_h

                for scale, ars in zip(self.scales, self.aspect_ratios):
                    for ar in ars:
                        ar = sqrt(ar)
                        w = scale * ar / 550
                        h = scale * ar / 550
                        prior_data += [x, y, w, h]

            self.priors = tf.reshape(tf.convert_to_tensor(prior_data), [-1, 4])
            self.last_conv_size = (conv_w, conv_h)

        return self.priors


class Yolact(tf.keras.Model):
    """


    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝


    You can set the arguments by chainging them in the backbone config object in config.py.

    Parameters (in cfg.backbone):
        - selected_layers: The indices of the conv layers to use for prediction.
        - pred_scales:     A list with len(selected_layers) containing tuples of scales (see PredictionModule)
        - pred_aspect_ratios: A list of lists of aspect ratios with len(selected_layers) (see PredictionModule)
    """

    def __init__(self):
        super(Yolact, self).__init__()

        self.backbone = darknet53
        # Compute mask_dim here and add it back to the config. Make sure Yolact's constructor is called early!
        self.num_grids = 0
        self.proto_src = 0

        in_channels = 256
        in_channels += self.num_grids

        self.proto_net = Protonet()
        self.selected_layers = [2, 3, 4]
        src_channels = self.backbone.channels

        # Some hacky rewiring to accomodate the FPN
        self.fpn = FPN([src_channels[i] for i in self.selected_layers])

        self.selected_layers = list(range(len(self.selected_layers) + 2))
        src_channels = [256] * len(self.selected_layers)

        self.prediction_layers = []
        pred_aspect_ratios = [ [[1, 1/2, 2]] ]*5
        pred_scales = [[24], [48], [96], [192], [384]]
        for idx, layer_idx in enumerate(self.selected_layers):
            # If we're sharing prediction module weights, have every module's parent be the first one
            parent = None
            if idx > 0: parent = self.prediction_layers[0]

            pred = PredictionModule(src_channels[layer_idx], src_channels[layer_idx],
                                    aspect_ratios = pred_aspect_ratios[idx],
                                    scales        = pred_scales[idx],
                                    parent        = parent)
            self.prediction_layers.append(pred)

        self.semantic_seg_conv = conv_layer(src_channels[0], 80, 1)

    def call(self, x):
        """ The input should be of size [batch_size, img_h, img_w, 3] """

        outs = self.backbone(x)
        # Use backbone.selected_layers because we overwrote self.selected_layers
        outs = [outs[i] for i in [2, 3, 4]]
        outs = self.fpn(outs)

        proto_out = None
        proto_x = x if self.proto_src is None else outs[self.proto_src]

        proto_out = self.proto_net(proto_x)
        proto_out = tf.nn.relu(proto_out)

        # Move the features last so the multiplication is easy
        pred_outs = { 'loc': [], 'conf': [], 'mask': [], 'priors': [] }

        for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
            pred_x = outs[idx]
            # A hack for the way dataparallel works
            if  pred_layer is not self.prediction_layers[0]:
                pred_layer.parent = [self.prediction_layers[0]]

            p = pred_layer(pred_x)
            for k, v in p.items():
                pred_outs[k].append(v)

        for k, v in pred_outs.items():
            pred_outs[k] = tf.concat(v, -2)

        if proto_out is not None: pred_outs['proto'] = proto_out
        pred_outs['conf'] = tf.nn.softmax(pred_outs['conf'], -1)

        return pred_outs



