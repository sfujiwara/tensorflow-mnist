# -*- coding: utf-8 -*-

import tensorflow as tf


class MnistSr:

    def __init__(self):
        pass

    @staticmethod
    def inference(x_ph):
        logits = tf.contrib.layers.fully_connected(x_ph, 10, activation_fn=None)
        outputs = tf.nn.softmax(logits)
        return outputs

    @staticmethod
    def build_loss(y_ph, logits):
        with tf.name_scope("loss"):
            cross_entropy = -tf.reduce_mean(y_ph * tf.log(logits))
        return cross_entropy
