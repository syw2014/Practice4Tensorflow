#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : model_fn.py
# PythonVersion: python3.5
# Date    : 2019/4/22 6:38
# Software: PyCharm

"""Model helper function."""
import tensorflow as tf


def build_model_spec(mode, model, inputs, params, reuse=False):
    """
    Build model specific
    :param mode: (string), run pattern,  `train`,`dev`
    :param model: (Object), model object
    :param inputs: (Dict), {'input_x': x_tensor, 'input_y': y_tensor}, defined in `input_fn`
    :param params: (Object),
    :param reuse:
    :return: (dict) model specification
    """
    model_spec = inputs
    is_training = (mode == 'train')
    sentences = inputs["sentence"]
    with tf.variable_scope("model", reuse=reuse):
        logits = model.inference(sentences, params, is_training)
        score = tf.nn.softmax(logits)
        predictions = tf.argmax(score, axis=1)

    if mode == "train":
        labels = inputs['labels']
        loss = model.loss(logits, labels)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

        # metrics and summaries
        # Metrics for evaluation using tf.metrics (average for whole dataset)
        with tf.variable_scope('metrics'):
            metrics = {
                'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions),
                'loss': tf.metrics.mean(loss)
            }
            # Group the update ops for the tf.metrics
        update_metrics_op = tf.group(*[op for _, op in metrics.values()])

        # Get the op to reset the local variables used in tf.metrics
        metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='metrics')
        metrics_init_op = tf.variables_initializer(metric_variables)

        # summaries for training
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)

        model_spec['loss'] = loss
        model_spec['accuracy'] = accuracy
        model_spec['metrics_init_op'] = metrics_init_op
        model_spec['metrics'] = metrics
        model_spec['update_metrics'] = update_metrics_op
        model_spec['summary_op'] = tf.summary.merge_all()
    elif mode == "eval":
        labels = inputs['labels']
        loss = model.loss(logits, labels)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

        # metrics and summaries
        # Metrics for evaluation using tf.metrics (average for whole dataset)
        with tf.variable_scope('metrics'):
            metrics = {
                'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions),
                'loss': tf.metrics.mean(loss)
            }
            # Group the update ops for the tf.metrics
        update_metrics_op = tf.group(*[op for _, op in metrics.values()])

        # Get the op to reset the local variables used in tf.metrics
        metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='metrics')
        metrics_init_op = tf.variables_initializer(metric_variables)

        # summaries for training
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)

        model_spec['loss'] = loss
        model_spec['accuracy'] = accuracy
        model_spec['metrics_init_op'] = metrics_init_op
        model_spec['metrics'] = metrics
        model_spec['update_metrics'] = update_metrics_op
        model_spec['summary_op'] = tf.summary.merge_all()

    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec['prediction'] = predictions
    model_spec['score'] = score

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec

