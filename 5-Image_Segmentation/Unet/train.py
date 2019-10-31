#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-09-19 15:25:10
#   Description :
#
#================================================================

import os
import cv2
import numpy as np
from Unet import Unet
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def DataGenerator(file_path, batch_size):
    """
    generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen
    to ensure the transformation for image and mask is the same
    """
    aug_dict = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')
    aug_dict = dict(horizontal_flip=True,
                        fill_mode='nearest')

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        file_path,
        classes=["images"],
        color_mode = "grayscale",
        target_size = (256, 256),
        class_mode = None,
        batch_size = batch_size, seed=1)

    mask_generator = mask_datagen.flow_from_directory(
        file_path,
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
        yield (img,mask)

model = Unet(1, image_size=256)
trainset = DataGenerator("membrane/train", batch_size=2)
model.fit_generator(trainset,steps_per_epoch=5000,epochs=5)
model.save_weights("model.h5")

testSet = DataGenerator("membrane/test", batch_size=1)
alpha   = 0.3
model.load_weights("model.h5")
if not os.path.exists("./results"): os.mkdir("./results")

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
    if idx == 29: break


