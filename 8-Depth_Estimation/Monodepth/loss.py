#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : loss.py
#   Author      : YunYang1994
#   Created date: 2019-11-02 10:48:03
#   Description :
#
#================================================================

import tensorflow as tf

from MonodepthNetwork import MonodepthNetwork
from utils import split_lr_disp, bilinear_sampler_1d_h, scale_pyramid, generate_left_image, generate_right_image

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = tf.nn.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = tf.nn.avg_pool2d(y, 3, 1, 'VALID')

    sigma_x  = tf.nn.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
    sigma_y  = tf.nn.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
    sigma_xy = tf.nn.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

def get_disparity_smoothness(disp, pyramid):
    gradient_x = lambda img: img[:,:,:-1,:] - img[:,:,1:,:]
    gradient_y = lambda img: img[:,:-1,:,:] - img[:,1:,:,:]

    disp_gradients_x = [gradient_x(d) for d in disp]
    disp_gradients_y = [gradient_y(d) for d in disp]

    image_gradients_x = [gradient_x(img) for img in pyramid]
    image_gradients_y = [gradient_y(img) for img in pyramid]

    weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keepdims=True)) for g in image_gradients_x]
    weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keepdims=True)) for g in image_gradients_y]

    smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
    smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
    return smoothness_x + smoothness_y

def compute_loss(left_image, right_image, lr_disp):
    true_left_pyramid  = scale_pyramid(left_image, 4)
    true_right_pyramid = scale_pyramid(right_image, 4)

    left_disp, right_disp = split_lr_disp(lr_disp)
    # generate images with disparity
    pred_left_pyramid  = [generate_left_image(true_right_pyramid[i], left_disp[i]) for i in range(4)]
    pred_right_pyramid = [generate_right_image(true_left_pyramid[i], right_disp[i]) for i in range(4)]

    # lr consitency
    right_to_left_disp= [generate_left_image(right_disp[i], left_disp[i]) for i in range(4)]
    left_to_right_disp= [generate_right_image(left_disp[i], right_disp[i]) for i in range(4)]

    # disparity smoothness
    disp_left_smoothness = get_disparity_smoothness(left_disp, true_left_pyramid)
    disp_right_smoothness = get_disparity_smoothness(right_disp, true_right_pyramid)

    # IMAGE RECONSTRUCTION
    # L1 loss
    l1_left = [tf.abs(pred_left_pyramid[i]-true_left_pyramid[i]) for i in range(4)]
    l1_reconstruction_loss_left = [tf.reduce_mean(l) for l in l1_left]
    l1_right = [tf.abs(pred_right_pyramid[i]-true_right_pyramid[i]) for i in range(4)]
    l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in l1_right]

    # SSIM loss
    ssim_left = [SSIM(pred_left_pyramid[i], true_left_pyramid[i]) for i in range(4)]
    ssim_loss_left = [tf.reduce_mean(s) for s in ssim_left]
    ssim_right = [SSIM(pred_right_pyramid[i], true_right_pyramid[i]) for i in range(4)]
    ssim_loss_right = [tf.reduce_mean(s) for s in ssim_right]

    # WEIGTHED SUM
    alpha = 0.85
    image_loss_right =[alpha*ssim_loss_right[i] + (1-alpha) * l1_reconstruction_loss_right[i] for i in range(4)]
    image_loss_left  =[alpha*ssim_loss_left[i] + (1-alpha) * l1_reconstruction_loss_left[i] for i in range(4)]
    image_loss = tf.add_n(image_loss_left + image_loss_right)

    # DISPARITY SMOOTHNESS
    disp_left_loss = [tf.reduce_mean(tf.abs(disp_left_smoothness[i])) / 2 ** i for i in range(4)]
    disp_right_loss = [tf.reduce_mean(tf.abs(disp_right_smoothness[i])) / 2 ** i for i in range(4)]
    disp_gradient_loss = 0.1 * tf.add_n(disp_left_loss + disp_right_loss)

    # LR CONSISTENCY
    lr_left_loss = [tf.reduce_mean(tf.abs(right_to_left_disp[i] - left_disp[i])) for i in range(4)]
    lr_right_loss = [tf.reduce_mean(tf.abs(left_to_right_disp[i] - right_disp[i])) for i in range(4)]
    lr_loss = tf.add_n(lr_left_loss + lr_right_loss)

    return image_loss, disp_gradient_loss, lr_loss


