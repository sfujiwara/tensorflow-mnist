# -*- coding: utf-8 -*-

import tensorflow as tf


class MnistCnn:

    def __init__(self):
        pass

    @staticmethod
    def inference(x_ph):
        # Resize rank 1 Tensor to rank 2 Tensor
        x_image_ph = tf.reshape(x_ph, [-1, 28, 28, 1])
        # Convolution layer 1
        h_conv1 = tf.contrib.layers.convolution2d(inputs=x_image_ph, num_outputs=32, kernel_size=5)
        h_pool1 = tf.contrib.layers.max_pool2d(h_conv1, kernel_size=[2, 2], stride=[2, 2], padding="SAME")
        # Convolution layer 2
        h_conv2 = tf.contrib.layers.convolution2d(inputs=h_pool1, num_outputs=64, kernel_size=5)
        h_pool2 = tf.contrib.layers.max_pool2d(h_conv2, kernel_size=[2, 2], stride=[2, 2], padding="SAME")
        # Flatten rank 2 Tensor to rank 1 Tensor
        h_pool2_flat = tf.contrib.layers.flatten(h_pool2)
        # Fully connected layer 1
        h_fc1 = tf.contrib.layers.fully_connected(h_pool2_flat, 1024)
        # Fully connected layer 2
        h_fc2 = tf.contrib.layers.fully_connected(h_fc1, 10, activation_fn=None)
        outputs = tf.nn.softmax(h_fc2)
        return outputs

    @staticmethod
    def build_loss(y_ph, logits):
        with tf.name_scope("loss"):
            cross_entropy = -tf.reduce_mean(y_ph * tf.log(logits))
        return cross_entropy

