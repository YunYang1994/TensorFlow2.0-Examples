#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-10-17 15:00:25
#   Description :
#
#================================================================

import os
import cv2
import random
import tensorflow as tf
import numpy as np
from utils import compute_iou, plot_boxes_on_image, load_gt_boxes, wandhG, compute_regression
from PIL import Image
from rpn import RPNplus

pos_thresh = 0.5
neg_thresh = 0.1
grid_width = grid_height = 16
image_height, image_width = 720, 960
synthetic_dataset_path="/Users/yangyun/synthetic_dataset"

def encode_label(image, gt_boxes):
    target_scores = np.zeros(shape=[45, 60, 9, 2]) # 0: background, 1: foreground, ,
    target_bboxes = np.zeros(shape=[45, 60, 9, 4]) # t_x, t_y, t_w, t_h
    target_masks  = np.zeros(shape=[45, 60, 9]) # negative_samples: -1, positive_samples: 1
    for i in range(45): # y: height
        for j in range(60): # x: width
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
                        plot_boxes_on_image(image, anchor_boxes, thickness=1)
                        print("=> encode: %d, %d, %d" %(i, j, k))
                        cv2.circle(image, center=(int(0.5*(xmin+xmax)), int(0.5*(ymin+ymax))),
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
                        cv2.circle(image, center=(int(0.5*(xmin+xmax)), int(0.5*(ymin+ymax))),
                                        radius=1, color=[0,0,0], thickness=4)
    Image.fromarray(image).show()
    return target_scores, target_bboxes, target_masks

def process_image_label(image_path, label_path):
    # image = Image.open(image_path)
    raw_image = cv2.imread(image_path)
    gt_boxes = load_gt_boxes(label_path)
    # show_image_with_boxes = np.copy(raw_image)
    # plot_boxes_on_image(show_image_with_boxes, gt_boxes)
    # Image.fromarray(show_image_with_boxes).show()

    show_image_with_posive_samples = np.copy(raw_image)
    target = encode_label(show_image_with_posive_samples, gt_boxes)
    return raw_image/255., target

def create_image_label_path_generator(synthetic_dataset_path):
    image_num = 8000
    image_num = 1
    image_label_paths = [(os.path.join(synthetic_dataset_path, "image/%d.jpg" %(idx+1)),
                          os.path.join(synthetic_dataset_path, "imageAno/%d.txt"%(idx+1))) for idx in range(image_num)]
    while True:
        random.shuffle(image_label_paths)
        for i in range(image_num):
            yield image_label_paths[i]

def DataGenerator(synthetic_dataset_path, batch_size):
    """
    generate image and mask at the same time
    """
    image_label_path_generator = create_image_label_path_generator(synthetic_dataset_path)
    while True:
        images = np.zeros(shape=[batch_size, image_height, image_width, 3], dtype=np.float)
        target_scores = np.zeros(shape=[batch_size, 45, 60, 9, 2], dtype=np.float)
        target_bboxes = np.zeros(shape=[batch_size, 45, 60, 9, 4], dtype=np.float)
        target_masks  = np.zeros(shape=[batch_size, 45, 60, 9], dtype=np.int)

        for i in range(batch_size):
            image_path, label_path = next(image_label_path_generator)
            print("=> ", image_path, " ", label_path)
            image, target = process_image_label(image_path, label_path)
            images[i] = image
            target_scores[i] = target[0]
            target_bboxes[i] = target[1]
            target_masks[i]  = target[2]
            yield images, target_scores, target_bboxes, target_masks


TrainSet = DataGenerator(synthetic_dataset_path, 1)
x, y1, y2, y3 = next(TrainSet)

# model = RPNplus()
# pred_y1, pred_y2 = model(x)

# score_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y1, logits=pred_y1)
# foreground_background_mask = (np.abs(y3) == 1).astype(np.int)
# score_loss = tf.reduce_sum(score_loss * foreground_background_mask, axis=[1,2,3]) / np.sum(foreground_background_mask)

# boxes_loss = tf.abs(y2 - pred_y2)
# boxes_loss = 0.5 * tf.pow(boxes_loss, 2) * tf.cast(boxes_loss<1, tf.float32) + (boxes_loss - 1) * tf.cast(boxes_loss >=1, tf.float32)
# boxes_loss = tf.reduce_sum(boxes_loss, axis=-1)
# foreground_mask = (y3 > 0).astype(np.float32)
# boxes_loss = tf.reduce_sum(boxes_loss * foreground_mask, axis=[1,2,3]) / np.sum(foreground_mask)


raw_image = x * 255.
raw_image = raw_image[0]
wandhG = np.array(wandhG)
score = y1
boxes = y2
score = tf.reshape(score, shape=[45, 60, 9, 2]).numpy()
boxes = tf.reshape(boxes, shape=[45, 60, 9, 4]).numpy()

pred_boxes = []
pred_score = []

for i in range(45):
    for j in range(60):
        for k in range(9):
            center_x = j * 16 + 8
            center_y = i * 16 + 8

            # 真实的 gt-boxes 坐标
            anchor_xmin = center_x - 0.5 * wandhG[k, 0]
            anchor_ymin = center_y - 0.5 * wandhG[k, 1]

            xmin = boxes[i, j, k, 0] * wandhG[k, 0] + anchor_xmin
            ymin = boxes[i, j, k, 1] * wandhG[k, 1] + anchor_ymin
            xmax = tf.exp(boxes[i, j, k, 2]) * wandhG[k, 0] + xmin
            ymax = tf.exp(boxes[i, j, k, 3]) * wandhG[k, 1] + ymin

            # anchor的实际坐标
            center_x = j * grid_width + grid_width * 0.5
            center_y = i * grid_height + grid_height * 0.5
            xmin = center_x - wandhG[k][0] * 0.5
            ymin = center_y - wandhG[k][1] * 0.5
            xmax = center_x + wandhG[k][0] * 0.5
            ymax = center_y + wandhG[k][1] * 0.5

            if score[i, j, k, 1] > 0:
                print("=> decode: %d, %d, %d" %(i, j, k))
                cv2.circle(raw_image, center=(int(0.5*(xmin+xmax)), int(0.5*(ymin+ymax))),
                                radius=1, color=[255,0,0], thickness=4)
                pred_boxes.append(np.array([xmin, ymin, xmax, ymax]))
                pred_score.append(score[i, j, k, 1])

pred_boxes = np.array(pred_boxes)
pred_score = np.array(pred_score)

# selected_boxes = pred_boxes
selected_boxes = []
while len(pred_boxes) > 0:
    max_idx = np.argmax(pred_score)
    selected_box = pred_boxes[max_idx]
    selected_boxes.append(selected_box)
    pred_boxes = np.concatenate([pred_boxes[:max_idx], pred_boxes[max_idx+1:]])
    pred_score = np.concatenate([pred_score[:max_idx], pred_score[max_idx+1:]])
    ious = compute_iou(selected_box, pred_boxes)
    iou_mask = ious <= 0.1
    pred_boxes = pred_boxes[iou_mask]
    pred_score = pred_score[iou_mask]

selected_boxes = np.array(selected_boxes)
plot_boxes_on_image(raw_image, selected_boxes)
Image.fromarray(np.uint8(raw_image)).show()


grid_size = [45, 60]

grid_x = tf.range(grid_size[0], dtype=tf.int32)
grid_y = tf.range(grid_size[1], dtype=tf.int32)
a, b = tf.meshgrid(grid_x, grid_y)
x_offset = tf.reshape(a, (-1, 1))
y_offset = tf.reshape(b, (-1, 1))
xy_offset = tf.concat([x_offset, y_offset], axis=-1)
xy_offset = tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2])
xy_offset = tf.cast(x_y_offset, tf.float32)

