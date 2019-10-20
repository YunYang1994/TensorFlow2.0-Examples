#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : demo.py
#   Author      : YunYang1994
#   Created date: 2019-10-20 15:06:46
#   Description :
#
#================================================================

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from rpn import RPNplus
from utils import compute_iou, plot_boxes_on_image, wandhG, load_gt_boxes, compute_regression

pos_thresh = 0.5
neg_thresh = 0.1
iou_thresh = 0.5
grid_width = grid_height = 16
image_height, image_width = 720, 960
wandhG = np.array(wandhG)

image_path = "/Users/yangyun/synthetic_dataset/image/1.jpg"
gt_boxes = load_gt_boxes("/Users/yangyun/synthetic_dataset/imageAno/1.txt")
raw_image = cv2.imread(image_path)
image_with_gt_boxes = np.copy(raw_image)
plot_boxes_on_image(image_with_gt_boxes, gt_boxes)
Image.fromarray(image_with_gt_boxes).show()
encoded_image = np.copy(raw_image)

target_scores = np.zeros(shape=[45, 60, 9, 2]) # 0: background, 1: foreground, ,
target_bboxes = np.zeros(shape=[45, 60, 9, 4]) # t_x, t_y, t_w, t_h
target_masks  = np.zeros(shape=[45, 60, 9]) # negative_samples: -1, positive_samples: 1

################################### ENCODE INPUT #################################

for i in range(45):
    for j in range(60):
        for k in range(9):
            center_x = j * grid_width + grid_width * 0.5
            center_y = i * grid_height + grid_height * 0.5
            xmin = center_x - wandhG[k][0] * 0.5
            ymin = center_y - wandhG[k][1] * 0.5
            xmax = center_x + wandhG[k][0] * 0.5
            ymax = center_y + wandhG[k][1] * 0.5
            # print(xmin, ymin, xmax, ymax)
            # ignore cross-boundary anchors
            if (xmin > -5) & (ymin > -5) & (xmax < (image_width+5)) & (ymax < (image_height+5)):
                anchor_boxes = np.array([xmin, ymin, xmax, ymax])
                anchor_boxes = np.expand_dims(anchor_boxes, axis=0)
                # compute iou between this anchor and all ground-truth boxes in image.
                ious = compute_iou(anchor_boxes, gt_boxes)
                positive_masks = ious > pos_thresh
                negative_masks = ious < neg_thresh

                if np.any(positive_masks):
                    plot_boxes_on_image(encoded_image, anchor_boxes, thickness=1)
                    print("=> Encoding positive sample: %d, %d, %d" %(i, j, k))
                    cv2.circle(encoded_image, center=(int(0.5*(xmin+xmax)), int(0.5*(ymin+ymax))),
                                    radius=1, color=[255,0,0], thickness=4)

                    target_scores[i, j, k, 1] = 1.
                    target_masks[i, j, k] = 1 # labeled as a positive sample
                    # find out which ground-truth box matches this anchor
                    max_iou_idx = np.argmax(ious)
                    selected_gt_boxes = gt_boxes[max_iou_idx]
                    target_bboxes[i, j, k] = compute_regression(selected_gt_boxes, anchor_boxes[0])

                if np.all(negative_masks):
                    target_scores[i, j, k, 0] = 1.
                    target_masks[i, j, k] = -1 # labeled as a negative sample
                    cv2.circle(encoded_image, center=(int(0.5*(xmin+xmax)), int(0.5*(ymin+ymax))),
                                    radius=1, color=[0,0,0], thickness=4)

Image.fromarray(encoded_image).show()

################################### DECODE OUTPUT #################################

decode_image = np.copy(raw_image)
pred_boxes = []
pred_score = []

for i in range(45):
    for j in range(60):
        for k in range(9):
            # 真实的 gt-boxes 坐标
            center_x = j * 16 + 8
            center_y = i * 16 + 8
            anchor_xmin = center_x - 0.5 * wandhG[k, 0]
            anchor_ymin = center_y - 0.5 * wandhG[k, 1]

            xmin = target_bboxes[i, j, k, 0] * wandhG[k, 0] + anchor_xmin
            ymin = target_bboxes[i, j, k, 1] * wandhG[k, 1] + anchor_ymin
            xmax = tf.exp(target_bboxes[i, j, k, 2]) * wandhG[k, 0] + xmin
            ymax = tf.exp(target_bboxes[i, j, k, 3]) * wandhG[k, 1] + ymin

            # # anchor的实际坐标
            # center_x = j * grid_width + grid_width * 0.5
            # center_y = i * grid_height + grid_height * 0.5
            # xmin = center_x - wandhG[k][0] * 0.5
            # ymin = center_y - wandhG[k][1] * 0.5
            # xmax = center_x + wandhG[k][0] * 0.5
            # ymax = center_y + wandhG[k][1] * 0.5

            if target_scores[i, j, k, 1] > 0:
                print("=> Decoding positive sample: %d, %d, %d" %(i, j, k))
                cv2.circle(decode_image, center=(int(0.5*(xmin+xmax)), int(0.5*(ymin+ymax))),
                                radius=1, color=[255,0,0], thickness=4)
                pred_boxes.append(np.array([xmin, ymin, xmax, ymax]))
                pred_score.append(target_scores[i, j, k, 1])

pred_boxes = np.array(pred_boxes)
plot_boxes_on_image(decode_image, pred_boxes)
Image.fromarray(np.uint8(decode_image)).show()





