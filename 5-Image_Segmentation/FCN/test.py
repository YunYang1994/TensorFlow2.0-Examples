#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : test.py
#   Author      : YunYang1994
#   Created date: 2019-10-23 23:14:38
#   Description :
#
#================================================================

import numpy as np
import tensorflow as tf

from fcn8s import FCN8s
from utils import visual_result, DataGenerator

model = FCN8s(n_class=21)
TestSet  = DataGenerator("./data/test_image.txt", "./data/test_labels", 1)

## load weights and test your model after training
## if you want to test model, first you need to initialize your model
## with "model(data)", and then load model weights
data = np.ones(shape=[1,224,224,3], dtype=np.float)
model(data)
model.load_weights("FCN8s.h5")

for idx, (x, y) in enumerate(TestSet):
    result = model(x)
    pred_label = tf.argmax(result, axis=-1)
    result = visual_result(x[0], pred_label[0].numpy())
    save_file = "./data/prediction/%d.jpg" %idx
    print("=> saving prediction result into ", save_file)
    result.save(save_file)
    if idx == 209:
        result.show()
        break

