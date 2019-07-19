#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : video_demo.py
#   Author      : YunYang1994
#   Created date: 2019-07-12 19:36:53
#   Description :
#
#================================================================

import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
import core.yolov3 as yolov3
from PIL import Image


video_path      = "./docs/road.mp4"
video_path      = 0
num_classes     = 80
input_size      = 416

input_layer = tf.keras.layers.Input([input_size, input_size, 3])
model = yolov3.YOLOV3(input_layer)
model.load_weights("./yolov3.weights")

vid = cv2.VideoCapture(video_path)
while True:
    return_value, frame = vid.read()
    if return_value:
        image = Image.fromarray(frame)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("No image!")
    frame_size = frame.shape[:2]
    image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    prev_time = time.time()

    pred_sbbox, pred_mbbox, pred_lbbox = model.inference(image_data)
    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

    bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
    bboxes = utils.nms(bboxes, 0.45, method='nms')
    image = utils.draw_bbox(frame, bboxes)

    curr_time = time.time()
    exec_time = curr_time - prev_time
    result = np.asarray(image)
    info = "time: %.2f ms" %(1000*exec_time)
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

