#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : train.py
# PythonVersion: python3.5
# Date    : 2019/4/22 下午6:48
# Software: PyCharm

"""Main train for project"""

import tensorflow as tf
from util_tools import Params, set_logger
from input_fn import load_dataset_from_text, input_fn
from model_fn import build_model_spec
from train_evaluate import train_and_evaluate, evaluate
from model import MyModel

import pickle
import argparse
import logging
import os


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='/root/data/research/data/product/feeds/dataset/output/base_model',
                    help='Directory containing params.json')
parser.add_argument('--data_dir', default='/root/data/research/data/product/feeds/dataset/',
                    help='Directory containing dataset')
parser.add_argument('--restore_dir', default=None,
                    help='Optional, directory containing weights to reload before training')


def main():
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load model parameters from params.json file in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No <params.json> json configuration file found at {}".format(args.model_dir)
    # load params
    params = Params(json_path)

    # Load dataset parameters from dataset_params.json file in data_dir
    json_path = os.path.join(args.data_dir, 'dataset_params.json')
    assert os.path.isfile(json_path),"No <dataset_params.json> json configuration file found at {}".format(args.data_dir)
    params.update(json_path)
    num_oov_buckets = params.num_oov_buckets # number of buckets for unknown words

    # Check we are not overwriting some previous results
    # if can comment this if you want to overwriting
    # model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, 'best_weights'))
    # overwritting = model_dir_has_best_weights and args.restore_dir is None
    # assert not overwritting, "Weights found in model dir, aborting to avoid overwrite"

    # Set logger
    set_logger(os.path.join(args.model_dir, 'train.log'))
    # print parameters
    params.print()

    # get vocabulary and label filename
    vocab_path = os.path.join(args.data_dir, 'words.txt')
    label_path = os.path.join(args.data_dir, 'tags.txt')

    train_sentences_path = os.path.join(args.data_dir, 'train/sentences.txt.list')
    train_labels_path = os.path.join(args.data_dir, 'train/labels.txt')

    eval_sentences_path = os.path.join(args.data_dir, 'dev/sentences.txt.list')
    eval_labels_path = os.path.join(args.data_dir, 'dev/labels.txt')

    # for batch predict
    # test_sentences_path  = os.path.join(args.data_dir, 'test/sentences.txt.list')
    # test_labels_path = os.path.join(args.data_dir, 'test/labels.txt')

    # Create word lookup table and label lookup table
    words_table = tf.contrib.lookup.index_table_from_file(vocab_path, num_oov_buckets=num_oov_buckets)
    tags_table = tf.contrib.lookup.index_table_from_file(label_path)

    # Create data input pipeline
    logging.info('Create dataset...')
    train_sentences = load_dataset_from_text(train_sentences_path, words_table)
    train_labels = load_dataset_from_text(train_labels_path, tags_table)

    eval_sentences = load_dataset_from_text(eval_sentences_path, words_table)
    eval_labels = load_dataset_from_text(eval_labels_path, tags_table)

    # Specify other parameters for the dataset and model
    params.eval_size = params.dev_size
    params.buffer_size = params.train_size # buffer size for shuffling, this will load all dataset into memory
    params.id_pad_word = words_table.lookup(tf.constant(params.pad_word))
    params.ld_pad_label = tags_table.lookup(tf.constant(params.pad_tag))

    # Create train and eval iterator over the two dataset
    train_inputs = input_fn('train', train_sentences, train_labels, params)
    eval_inputs = input_fn('eval', eval_sentences, eval_labels, params)
    logging.info("- Done")

    # Define the models (two different set of nodes that share weights for train and eval)
    logging.info("Creating the model...")
    model = MyModel()

    train_model_spec = build_model_spec('train', model, train_inputs, params)
    # IF you want to only run evaluate please set resue=False, when training you should reuse variables
    # eval_model_spec = model_fn('eval', model, eval_inputs, params, reuse=True)
    eval_model_spec = build_model_spec('eval', model, eval_inputs, params, reuse=True)
    logging.info('- Done.')

    # Train the model
    logging.info("Starting training for {} epochs".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params,
                       args.restore_dir)

    # write model as pickle for inference
    with open(args.model_dir + "/mymodel.pkl", 'wb') as fout:
        pickle.dump(model, fout)

    # evaluate
    # evaluate(eval_model_spec, args.model_dir, params, args.restore_dir)


if __name__ == "__main__":
    main()