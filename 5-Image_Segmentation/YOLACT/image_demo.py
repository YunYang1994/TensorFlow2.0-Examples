#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-07-29 11:55:07
#   Description :
#
#================================================================

import cv2
import numpy as np
import tensorflow as tf
from load_weights import model

image = cv2.imread("./boy.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (550, 550))
image = image / 255.
image = np.expand_dims(image, 0)


result = model.predict(image)





