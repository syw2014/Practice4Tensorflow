#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : train_evaluate.py
# PythonVersion: python3.5
# Date    : 2019/3/28 11:06
# Software: PyCharm

"""Define train and evaluate graph and sess"""

import logging
import os
import tensorflow as tf
from util_tools import save_dict_json
from tqdm import trange
import numpy as np


def train_sess(sess, model_spec, num_steps, writer, params):
    """
    Define train graph
    :param sess: tf.Session(), current session
    :param model_spec: (dict) contains the graph operations or nodes needed for training
    :param num_steps: (int) train for this number of batches
    :param writer: (tf.summary.FileWriter) writer for summaries
    :param params: (Params) hyperparameters
    :return:
    """
    # Get relevant graph operations or nodes needed for training
    loss = model_spec['loss']
    train_op = model_spec['train_op']
    update_metrics = model_spec['update_metrics']
    metrics = model_spec['metrics']
    summary_op = model_spec['summary_op']
    global_step = tf.train.get_or_create_global_step()

    # load the training dataset into pipeline and initialize the metrics local variables
    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    # use tqdm for progress bar
    t = trange(num_steps)
    for i in t:
        # evaluate summaries for tensorboard only once a while
        if i % params.save_summary_steps == 0:
            # perform a mini-batch update
            _, _, loss_val, summ, global_step_val = sess.run([train_op, update_metrics, loss,
                                                              summary_op, global_step])
            # write to summary
            writer.add_summary(summ, global_step_val)
        else:
            _, _, loss_val = sess.run([train_op, update_metrics, loss])
            t.set_postfix(loss='{:05.3f}'.format(loss_val))

    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k,v) for k, v in metrics_val.items())
    logging.info("- Train metrics: " + metrics_string)


def evaluate_sess(sess, model_spec, num_steps, writer=None, params=None):
    """
    Train the model on `num_steps` batches
    :param sess: tf.Session, current session
    :param model_spec: dict, contains the grap operations or nodes needed for training
    :param num_steps: int, train for this number of batches
    :param writer: tf.summary.FileWriter, writer for summaries. Is none log nothing
    :param params: (Params) hyperparameters
    :return:
    """
    # get ops
    update_metrics = model_spec['update_metrics']
    eval_metrics = model_spec['metrics']
    global_step = tf.train.get_global_step()

    # load the evaluation dataset into pipeline and initialize the metrics op
    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    # compute metrics over all dataset
    for _ in range(num_steps):
        sess.run(update_metrics)

    # get values tensor of metrics
    metrics_values = {k: v[0] for k, v in eval_metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Eval metrics: " + metrics_string)

    # add summaries manually to writer at gloabl_step_val
    if writer is not None:
        global_step_val = sess.run(global_step)
        for tag, val in metrics_val.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
            writer.add_summary(summ, global_step_val)
    return metrics_val


def evaluate(mode_spec, model_dir, params, restore_from):
    """
    Evaluate model
    :param mode_spec: dict, contains the graph operations or nodes needed for evaluation
    :param model_dir: string ,directory containing config, weights and log
    :param params: Params, hyperparameters
    :param restore_from: string, directory or file containing weights to restore the graph
    :return:
    """
    # initialize the tf.Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # initialize the lookup table
        sess.run(mode_spec['variable_init_op'])

        # restore weights form the weights subdirectory
        save_path = os.path.join(model_dir, restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)

        # Evaluate
        num_steps = (params.dev_size + params.batch_size - 1) // params.batch_size
        metrics = evaluate_sess(sess, mode_spec, num_steps)
        metrics_name = '_'.join(restore_from.split('/'))
        save_path = os.path.join(model_dir, "metrics_test_{}.json".format(metrics_name))
        save_dict_json(metrics, save_path)


def train_and_evaluate(train_model_spec, eval_model_spec, model_dir, params, restore_from=None):
    """
    Train the model and evaluate model in every epoch
    :param train_model_spec: dict, contains the graph operations or nodes needed for training
    :param eval_model_spec: dict, contains the graph operations or nodes needed for evaluation
    :param model_dir: string, directory containing config, weights and log
    :param params: Params, contains hyperparameters of the model
        Must define, num_epochs, train_size, batch_size, eval_size, save_summary_steps
    :param restore_from: string, directory or file containing weights to restore the graph
    :return:
    """
    # Initialize tf.Saver instances to save weights during training
    last_saver = tf.train.Saver()   # will keep last 5 eppochs
    best_saver = tf.train.Saver(max_to_keep=1)  # only keep 1 best checkpoint (best on eval)
    begin_at_epoch = 0

    with tf.Session() as sess:
        # Initialize variables
        sess.run(train_model_spec['variable_init_op'])

        # Reload weights from directory if specified
        if restore_from is not None:
            logging.info("Restoring parameters from {}".format(restore_from))
            # save_path = os.path.join(model_dir, restore_from)
            if os.path.isdir(restore_from):
                restore_from = tf.train.latest_checkpoint(restore_from)
                begin_at_epoch = int(restore_from.split('-')[-1])
            last_saver.restore(sess, restore_from)

        # For tensorboard (takes care of writing summaries to files)
        train_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train_summaries'), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(model_dir, 'eval_summaries'), sess.graph)

        best_eval_acc = 0.0
        for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs):
            # Run on epoch
            logging.info("Epoch {}/{}".format(epoch+1, begin_at_epoch + params.num_epochs))
            # compute number of batches in one epochs(one full pass over the training set)
            num_steps = (params.train_size + params.batch_size - 1) // params.batch_size
            train_sess(sess, train_model_spec, num_steps, train_writer, params)

            # Save weights
            last_save_path = os.path.join(model_dir, 'last_weights', 'after-epoch')
            last_saver.save(sess, last_save_path, global_step=epoch+1)

            # Evaluate for one epoch on validation set
            num_steps = (params.eval_size + params.batch_size - 1) // params.batch_size
            metrics = evaluate_sess(sess, eval_model_spec, num_steps, eval_writer)

            # If best eval, best save path
            eval_acc = metrics['accuracy']
            if eval_acc >= best_eval_acc:
                # store new best accuracy
                best_eval_acc = eval_acc
                # save weights
                best_save_path = os.path.join(model_dir, 'best_weights', 'after-epoch')
                best_save_path = best_saver.save(sess, best_save_path, global_step=epoch+1)

                logging.info("- Found new best accuracy, saving in {}".format(best_save_path))
                # Save best eval metrics in a json file in the model directory
                best_json_path = os.path.join(model_dir, 'metrics_eval_best_weights.json')
                save_dict_json(metrics, best_json_path)

            # Save latest eval metrics in a json file in the model directory
            latest_json_path = os.path.join(model_dir, 'metrics_eval_last_weights.json')
            save_dict_json(metrics, latest_json_path)