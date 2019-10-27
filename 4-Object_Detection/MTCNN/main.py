#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : main.py
#   Author      : YunYang1994
#   Created date: 2019-10-27 12:51:40
#   Description :
#
#================================================================

import cv2
import numpy as np
import tensorflow as tf
from mtcnn import PNet, RNet, ONet
from PIL import Image, ImageDraw
from utils import detect_face

def load_weights(model, weights_file):
    weights_dict = np.load(weights_file, encoding='latin1').item()
    for layer_name in weights_dict.keys():
        layer = model.get_layer(layer_name)
        if "conv" in layer_name:
            layer.set_weights([weights_dict[layer_name]["weights"], weights_dict[layer_name]["biases"]])
        else:
            prelu_weight = weights_dict[layer_name]['alpha']
            try:
                layer.set_weights([prelu_weight])
            except:
                layer.set_weights([prelu_weight[np.newaxis, np.newaxis, :]])
    return True

pnet, rnet, onet = PNet(), RNet(), ONet()
pnet(tf.ones(shape=[1, 224, 224, 3]))
rnet(tf.ones(shape=[1,  24,  24 ,3]))
onet(tf.ones(shape=[1,  48,  48, 3]))
load_weights(pnet, "./det1.npy"), load_weights(rnet, "./det2.npy"), load_weights(onet, "./det3.npy")

image = cv2.cvtColor(cv2.imread("./multiface.jpg"), cv2.COLOR_BGR2RGB)
total_boxes, points = detect_face(image, 20, pnet, rnet, onet, [0.6, 0.7, 0.7], 0.709)

for bounding_box, keypoints in zip(total_boxes, points.T):
    bounding_boxes = {
            'box': [int(bounding_box[0]), int(bounding_box[1]),
                    int(bounding_box[2]-bounding_box[0]), int(bounding_box[3]-bounding_box[1])],
            'confidence': bounding_box[-1],
            'keypoints': {
                'left_eye': (int(keypoints[0]), int(keypoints[5])),
                'right_eye': (int(keypoints[1]), int(keypoints[6])),
                'nose': (int(keypoints[2]), int(keypoints[7])),
                'mouth_left': (int(keypoints[3]), int(keypoints[8])),
                'mouth_right': (int(keypoints[4]), int(keypoints[9])),
            }
        }
    bounding_box = bounding_boxes['box']
    keypoints = bounding_boxes['keypoints']
    cv2.rectangle(image,
                (bounding_box[0], bounding_box[1]),
                (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                (0,155,255), 2)
    cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

Image.fromarray(image).show()
