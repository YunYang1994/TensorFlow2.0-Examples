#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : test.py
#   Author      : YunYang1994
#   Created date: 2019-10-12 17:45:33
#   Description :
#
#================================================================

import numpy as np
from fcn8s import FCN8s

data = np.arange(224*224*3).reshape([1, 224,224,3]).astype(np.float)

model = FCN8s()
y = model(data)
