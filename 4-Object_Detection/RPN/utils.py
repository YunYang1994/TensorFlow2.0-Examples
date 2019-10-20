#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : utils.py
#   Author      : YunYang1994
#   Created date: 2019-10-19 23:04:24
#   Description :
#
#================================================================

import cv2
import numpy as np

wandhG = [[ 74., 149.],
          [ 34., 149.],
          [ 86.,  74.],
          [109., 132.],
          [172., 183.],
          [103., 229.],
          [149.,  91.],
          [ 51., 132.],
          [ 57., 200.]]

def compute_iou(boxes1, boxes2):
    """(xmin, ymin, xmax, ymax)
    boxes1 shape:  [-1, 4], boxes2 shape: [-1, 4]
    """
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2], )
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_wh = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    union_area = boxes1_area + boxes2_area - inter_area
    ious = inter_area / union_area
    return ious

def plot_boxes_on_image(show_image_with_boxes, boxes, color=[0, 0, 255], thickness=2):
    for box in boxes:
        cv2.rectangle(show_image_with_boxes,
                pt1=(int(box[0]), int(box[1])),
                pt2=(int(box[2]), int(box[3])), color=color, thickness=thickness)
    # show_image_with_boxes = cv2.cvtColor(show_image_with_boxes, cv2.COLOR_BGR2RGB)

def load_gt_boxes(path):
    """
    Don't care about what shit it is. whatever, this function
    returns many ground truth boxes with the shape of [-1, 4].

    xmin, ymin, xmax, ymax
    """
    bbs = open(path).readlines()[1:]
    roi = np.zeros([len(bbs), 4])
    for iter_, bb in zip(range(len(bbs)), bbs):
        bb = bb.replace('\n', '').split(' ')
        bbtype = bb[0]
        bba = np.array([float(bb[i]) for i in range(1, 5)])
        occ = float(bb[5])
        bbv = np.array([float(bb[i]) for i in range(6, 10)])
        ignore = int(bb[10])

        ignore = ignore or (bbtype != 'person')
        ignore = ignore or (bba[3] < 40)
        bba[2] += bba[0]
        bba[3] += bba[1]

        roi[iter_, :4] = bba
    return roi

def compute_regression(box1, box2):
    """
    box1: ground-truth boxes
    box2: anchor boxes
    """
    target_reg = np.zeros(shape=[4,])
    w1 = box1[2] - box1[0]
    h1 = box1[3] - box1[1]
    w2 = box2[2] - box2[0]
    h2 = box2[3] - box2[1]

    target_reg[0] = (box1[0] - box2[0]) / w2
    target_reg[1] = (box1[1] - box2[1]) / h2
    target_reg[2] = np.log(w1 / w2)
    target_reg[3] = np.log(h1 / h2)

    return target_reg

