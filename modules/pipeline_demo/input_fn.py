#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : input_fn.py
# PythonVersion: python3.5
# Date    : 2019/4/21 下午7:23
# Software: PyCharm

"""Create input data pipeline with tf.dataset api"""
import tensorflow as tf
from util_tools import Params


def pad(seq_id_tensor, params):
    """
    Pad input sequence tensor to specific sequence length.
    :param seq_id_tensor: tensor, shape=[id1, id2, id3,...]
    :return: padded tensor
    """
    # if is_label_pad:
    #     return id_seq_tensor
    # create word ids padding
    padding = tf.constant([[0, 0], [0, params.max_doc_len]], dtype=tf.int32)

    # pad all the word ids entries to the maximum document length
    word_ids_padded = tf.pad([seq_id_tensor], padding)
    word_id_vector = tf.slice(word_ids_padded, [0, 0], [-1, params.max_doc_len])

    # convert data type as your model need
    tf.cast(word_id_vector, tf.int32)
    return tf.squeeze(word_id_vector, [0])


def load_dataset_from_text(file_name, lookupTable):
    """
    Create tf.data instance from input file
    :param file_name: input file name or path, each line was an example/senteces.
    :param lookupTable: tf.lookuptable
    :param Parameters
    :return: yeild list of ids of tokens for each example
    """
    # load txt file , one example per line
    dataset = tf.data.TextLineDataset(file_name)

    # convert line into list of tokens, splitting by white space, Notes,I had tokenize sentences to chars
    # you also can use words
    # tf.string_split, return a sparse tensor, so we only need values
    dataset = dataset.map(
        lambda string: tf.string_split([string], delimiter=' ').values)

    # Lookup table, convert text char/word to their ids, dataset content ids and it's length
    dataset = dataset.map(
        lambda tokens: lookupTable.lookup(tokens))

    return dataset


def input_fn(mode, sentences, labels, params):
    """
    Input function for classify
    :param mode: string, train or eval
    :param sentences: tf.Dataset, yielding list of ids of words
    :param labels: tf.Dataset, yielding list of ids of tags
    :param params: Parameters
    :return: (dict),
    """

    is_training = (mode == 'train')
    # Load all the dataset in memory for shuffling is training
    buffer_size = params.buffer_size if is_training else 1

    # padding
    sentences = sentences.map(
        lambda tokens: pad(tokens, params)
    )

    # Reshape of labels
    labels = labels.map(lambda label: tf.squeeze(label))

    # zip the sentences dan labels together
    dataset = tf.data.Dataset.zip((sentences, labels))

    # TODO, padding batch here
    # padded_shapes = ([None], 1)   # used for padded batch, every length has the same length
    # create batches and pad the sentences of different sentences, only pad sentences
    dataset = (dataset.shuffle(buffer_size=buffer_size)
               # .padded_batch(params.batch_size, padded_shapes=padded_shapes)
            .batch(params.batch_size)
               .prefetch(1)     # make sure always have one batch ready to serve
               )
    # create initializable iterator form this dataset so that we can reset at each epoch
    iterator = dataset.make_initializable_iterator()
    (features, label) = iterator.get_next()
    init_op = iterator.initializer

    # build and return a dictionary containing the nodes / ops

    inputs = {
        'sentence': features,   # [batch_size, sequence_len]
        'labels': label,        # [batch_size, 1] == [batch_size]
        'iterator_init_op': init_op
    }

    return inputs


if __name__ == '__main__':

    DIR = "/root/data/research/data/product/feeds/dataset/"
    params = Params(DIR+'dataset_params.json')
    params.buffer_size = params.test_size
    params.batch_size = 4
    params.max_doc_len = 10

    # lookup tabels
    word_path = DIR + 'words.txt'
    words_table = tf.contrib.lookup.index_table_from_file(word_path, num_oov_buckets=1)
    label_table = tf.contrib.lookup.index_table_from_file(DIR+'labels.txt')

    # dataset
    test_sentences = load_dataset_from_text(DIR + 'test/sentences.txt.list', words_table)
    test_labels = load_dataset_from_text(DIR + 'test/labels.txt', label_table)
    #
    # # input iterator
    test_inputs = input_fn('train', test_sentences, test_labels, params)

    with tf.Session() as sess:
        sess.run(test_inputs['iterator_init_op'])
        sess.run(tf.tables_initializer())
        print("x shape: ", test_inputs['sentence'].get_shape)
        print("y shape: ", test_inputs['labels'].get_shape)
        print(sess.run(test_inputs['sentence']))
        print(sess.run(test_inputs['labels']))