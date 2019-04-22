#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : model.py
# PythonVersion: python3.6
# Date    : 2019/421 下午8:10
# IDE     : PyCharm

import tensorflow as tf


class MyModel:
    """
    Define Deep Models, here we only define two function `inference()` and `loss()`, why design like this,as we want to
    use tensorflow high level api when training and evalute like tf.data.dataset api,but it difficult to use when
    inference with single sample.
    `inference`: get network outputs(logits), used in train and eval with tf.data.dataset api or other data iterator
    `loss`: compute model loss
    """
    @staticmethod
    def inference(sentence, params, is_training=False):
        """
        Portion of the compute graph that takes as input and converts it to a Y output(logit)
        :param sentence: input tensor, shape=[batch_size, dim]
        :param params: (Object)Param, a dictionary of parameters
        :return:
        """

        # build graph
        # with tf.device('/cpu:0'):
        embeddings = tf.get_variable(name='embeddings', dtype=tf.float32,
                                     shape=[params.vocab_size, params.embedding_size])
        sentence = tf.nn.embedding_lookup(embeddings, sentence)

        conv = tf.layers.conv1d(sentence, params.num_filters, params.kernel_size, name='conv')
        # global max pooling
        gmp = tf.reduce_mean(conv, reduction_indices=[1], name='max_pooling')

        fc = tf.layers.dense(gmp, params.hidden_dim, name='fc1')
        fc = tf.layers.dropout(fc, params.dropout_rate)
        fc = tf.nn.relu(fc)

        logits = tf.layers.dense(fc, params.num_class)

        return logits

    @staticmethod
    def loss(logits, labels):
        """
        Compute model loss, also design loss here
        :param logits: network ouputs tenor, shape=[batch, num_class_dim]
        :param labels: target label tensor, shape=[batch, label_ids] or shape=[batch, label_encoding_dim]
        :return:  loss tensor, scalar
        """
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss_mean = tf.reduce_mean(losses)
        return loss_mean