#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : test.py
#   Author      : YunYang1994
#   Created date: 2019-10-19 17:21:54
#   Description :
#
#================================================================

import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from rpn import RPNplus
from utils import decode_output, plot_boxes_on_image, nms

synthetic_dataset_path ="./synthetic_dataset"
prediction_result_path = "./prediction"
if not os.path.exists(prediction_result_path): os.mkdir(prediction_result_path)

model = RPNplus()
fake_data = np.ones(shape=[1, 720, 960, 3]).astype(np.float32)
model(fake_data) # initialize model to load weights
model.load_weights("./RPN.h5")

for idx in range(8000, 8200):
    image_path = os.path.join(synthetic_dataset_path, "image/%d.jpg" %(idx+1))
    raw_image = cv2.imread(image_path)
    image_data = np.expand_dims(raw_image / 255., 0)
    pred_scores, pred_bboxes = model(image_data)
    pred_scores = tf.nn.softmax(pred_scores, axis=-1)
    pred_scores, pred_bboxes = decode_output(pred_bboxes, pred_scores, 0.9)
    pred_bboxes = nms(pred_bboxes, pred_scores, 0.5)
    plot_boxes_on_image(raw_image, pred_bboxes)
    save_path = os.path.join(prediction_result_path, str(idx)+".jpg")
    print("=> saving prediction results into %s" %save_path)
    Image.fromarray(raw_image).save(save_path)

