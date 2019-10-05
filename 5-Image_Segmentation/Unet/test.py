#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : test.py
#   Author      : YunYang1994
#   Created date: 2019-10-05 12:55:29
#   Description :
#
#================================================================

import cv2
import numpy as np
from Unet import Unet
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def testGenerator(batch_size):
    """
    generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen
    to ensure the transformation for image and mask is the same
    """
    aug_dict = dict(horizontal_flip=False,fill_mode='nearest')

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        "membrane/test",
        classes=["images"],
        color_mode = "grayscale",
        target_size = (256, 256),
        class_mode = None,
        batch_size = batch_size, seed=1)

    mask_generator = mask_datagen.flow_from_directory(
        "membrane/test",
        classes=["labels"],
        color_mode = "grayscale",
        target_size = (256, 256),
        class_mode = None,
        batch_size = batch_size, seed=1)

    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img = img / 255.
        mask = mask / 255.
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        yield (img, mask[0])

testSet = testGenerator(batch_size=1)
alpha   = 0.3
model   = Unet(1, image_size=256)
model.load_weights("model.hf5")

for idx, (img, mask) in enumerate(testSet):
    oring_img = img[0]
    pred_mask = model.predict(img)[0]
    pred_mask[pred_mask > 0.5] = 1
    pred_mask[pred_mask <= 0.5] = 0
    img = cv2.cvtColor(img[0], cv2.COLOR_GRAY2RGB)
    H, W, C = img.shape
    for i in range(H):
        for j in range(W):
            if pred_mask[i][j][0] <= 0.5:
                img[i][j] = (1-alpha)*img[i][j]*255 + alpha*np.array([0, 0, 255])
            else:
                img[i][j] = img[i][j]*255
    image_accuracy = np.mean(mask == pred_mask)
    image_path = "./results/pred_"+str(idx)+".png"
    print("=> accuracy: %.4f, saving %s" %(image_accuracy, image_path))
    cv2.imwrite(image_path, img)
    cv2.imwrite("./results/origin_%d.png" %idx, oring_img*255)
    cv2.imwrite("./results/gt_%d.png" %idx, mask*255)
    if idx == 1: break


