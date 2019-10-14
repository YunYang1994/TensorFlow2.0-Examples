#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : Resnet.py
#   Author      : YunYang1994
#   Created date: 2019-10-11 19:16:55
#   Description :
#
#================================================================

import tensorflow as tf

# 3x3 convolution
def conv3x3(in_channels, out_channels, strides=1):
    """
    Implements a convolution layer.
    Arguments are passed into the conv layer.
    """
    x = tf.keras.layers.Input([None, None, in_channels])
    y = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3,
                               strides=strides, padding="same", use_bias=False)(x)
    return tf.keras.Model(x,y)

# Residual block
class ResidualBlock(tf.keras.Model):
    def __init__(self, in_channels, out_channels, strides=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, strides)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.downsample = downsample

    def call(self, x, training=False):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = tf.nn.relu(out)
        return out

# Resnet
class ResNet(tf.keras.Model):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = tf.keras.layers.BatchNormalization()
        self.layer1 = self.make_layer(block, 16, layers[0], 1)
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = tf.keras.layers.AveragePooling2D(8)
        self.fc = tf.keras.layers.Dense(units=num_classes)

    def make_layer(self, block, out_channels, blocks, strides=1):
        downsample = None
        if (strides != 1) or (self.in_channels != out_channels):
            downsample = tf.keras.Sequential([
                conv3x3(self.in_channels, out_channels, strides),
                tf.keras.layers.BatchNormalization()]
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, strides, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return tf.keras.Sequential(layers)

    def call(self, x, training=False):
        out = self.conv(x)
        out = self.bn(out, training=training)
        out = tf.nn.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = tf.reshape(out, (out.shape[0], -1))
        out = self.fc(out)
        return tf.nn.softmax(out, axis=-1)

# Load and prepare the cifar10 dataset.
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = tf.reshape(y_train, (-1,)), tf.reshape(y_test, (-1,))

# Use tf.data to batch and shuffle the dataset
train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(32)

model = ResNet(ResidualBlock, [2, 2, 2])

# Choose an optimizer and loss function for training
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Select metrics to measure the loss and the accuracy of the model
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# Use tf.GradientTape to train the model.
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        print("=> label shape: ", labels.shape, "pred shape", predictions.shape)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCHS = 5
for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = '=> Epoch {}, Loss: {:.4}, Accuracy: {:.2%}, Test Loss: {:.4}, Test Accuracy: {:.2%}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result(),
                          test_loss.result(),
                          test_accuracy.result()))
    # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()


