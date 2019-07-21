#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-07-12 13:07:27
#   Description :
#
#================================================================

import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode
from PIL import Image
from core.config import cfg

input_size   = 416
NUM_CLASS    = len(utils.read_class_names(cfg.YOLO.CLASSES))
image_path   = "./docs/kite.jpg"

input_layer  = tf.keras.layers.Input([input_size, input_size, 3])
feature_maps = YOLOv3(input_layer)

original_image      = cv2.imread(image_path)
original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]

image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...].astype(np.float32)

bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(tf.reshape(bbox_tensor, (-1, 5+NUM_CLASS)))

bbox_tensors = tf.concat(bbox_tensors, axis=0)
model = tf.keras.Model(input_layer, bbox_tensors)
utils.load_weights(model, "./yolov3.weights")
model.summary()

pred_bbox = model.predict(image_data)
bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
bboxes = utils.nms(bboxes, 0.45, method='nms')

image = utils.draw_bbox(original_image, bboxes)
image = Image.fromarray(image)
image.show()


