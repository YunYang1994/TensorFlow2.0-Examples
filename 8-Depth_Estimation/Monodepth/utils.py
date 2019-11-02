#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : utils.py
#   Author      : YunYang1994
#   Created date: 2019-11-02 10:52:29
#   Description :
#
#================================================================

import tensorflow as tf

def split_lr_disp(lr_disp):
    left_disp_pyramid  = [tf.expand_dims(d[:, :, :, 0], axis=3) for d in lr_disp]
    right_disp_pyramid = [tf.expand_dims(d[:, :, :, 1], axis=3) for d in lr_disp]
    return left_disp_pyramid, right_disp_pyramid

def bilinear_sampler_1d_h(input_images, x_offset, wrap_mode='border', name='bilinear_sampler', **kwargs):
    def _repeat(x, n_repeats):
        rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
        return tf.reshape(rep, [-1])

    def _interpolate(im, x, y):
        # handle both texture border types
        _edge_size = 0
        if _wrap_mode == 'border':
            _edge_size = 1
            im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
            x = x + _edge_size
            y = y + _edge_size
        elif _wrap_mode == 'edge':
            _edge_size = 0
        else:
            return None

        x = tf.clip_by_value(x, 0.0,  _width_f - 1 + 2 * _edge_size)

        x0_f = tf.floor(x)
        y0_f = tf.floor(y)
        x1_f = x0_f + 1

        x0 = tf.cast(x0_f, tf.int32)
        y0 = tf.cast(y0_f, tf.int32)
        x1 = tf.cast(tf.minimum(x1_f,  _width_f - 1 + 2 * _edge_size), tf.int32)

        dim2 = (_width + 2 * _edge_size)
        dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
        base = _repeat(tf.range(_num_batch) * dim1, _height * _width)
        base_y0 = base + y0 * dim2
        idx_l = base_y0 + x0
        idx_r = base_y0 + x1

        im_flat = tf.reshape(im, tf.stack([-1, _num_channels]))

        pix_l = tf.gather(im_flat, idx_l)
        pix_r = tf.gather(im_flat, idx_r)

        weight_l = tf.expand_dims(x1_f - x, 1)
        weight_r = tf.expand_dims(x - x0_f, 1)

        return weight_l * pix_l + weight_r * pix_r

    def _transform(input_images, x_offset):
        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        x_t, y_t = tf.meshgrid(tf.linspace(0.0,   _width_f - 1.0,  _width),
                                tf.linspace(0.0 , _height_f - 1.0 , _height))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
        y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

        x_t_flat = tf.reshape(x_t_flat, [-1])
        y_t_flat = tf.reshape(y_t_flat, [-1])

        x_t_flat = x_t_flat + tf.reshape(x_offset, [-1]) * _width_f

        input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

        output = tf.reshape(
            input_transformed, tf.stack([_num_batch, _height, _width, _num_channels]))
        return output

    _num_batch    = tf.shape(input_images)[0]
    _height       = tf.shape(input_images)[1]
    _width        = tf.shape(input_images)[2]
    _num_channels = tf.shape(input_images)[3]

    _height_f = tf.cast(_height, tf.float32)
    _width_f  = tf.cast(_width,  tf.float32)

    _wrap_mode = wrap_mode

    output = _transform(input_images, x_offset)
    return output

def generate_left_image(right_image, disp):
    return bilinear_sampler_1d_h(right_image, -disp)

def generate_right_image(left_image, disp):
    return bilinear_sampler_1d_h(left_image, disp)

def upsample_nn(image, scale):
    b, h, w, c = image.shape
    out = tf.image.resize(image, [int(h * scale), int(w * scale)], method='nearest')
    return out

def scale_pyramid(image, num_scales):
    scaled_images = [image]
    for i in range(num_scales-1):
        scale = 2 ** (i+1)
        scale = 1 / scale
        scaled_images.append(upsample_nn(image, scale))
    return scaled_images


