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

from Unet import Unet
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def trainGenerator(batch_size):
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
        "membrane/train",
        classes=["images"],
        color_mode = "grayscale",
        target_size = (256, 256),
        class_mode = None,
        batch_size = batch_size, seed=1)

    mask_generator = mask_datagen.flow_from_directory(
        "membrane/train",
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

trainset = trainGenerator(batch_size=2)
model = Unet(1, image_size=256)
model.fit_generator(trainset,steps_per_epoch=2000,epochs=5)
model.save_weights("model.h5")


