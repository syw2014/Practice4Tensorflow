#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : dataset_api_demo.py
# PythonVersion: python3.5
# Date    : 2019/3/21 16:19
# Software: PyCharm

"""A demo to show how to use tf.dataset api for NLP tasks. Different from many other tutorials."""

import tensorflow as tf
import multiprocessing

# define hyper-parameter
HEADER = ["label", "text"]
LABEL = 'label'
MAX_DOC_LEN = 30
NUM_CLASS = 7
PAD_WORD = "UNK"
DIR = "/data/research/data/product/classification/"

# define label text for corpus
LABELS = ['动漫', '短视频', '美图', '新闻', '影视', '游戏', '直播']


def decode_csv(line):
    """
    Parse each line in tf.dataset.map process
    :param line: string tensor
    :return: prased feature string tensor and label string tensor
    """
    # Note: here the column in our corpus only has two fields, label and token string, so record_defaults only has
    # two default value,both of them are string type, your should change the default values according your dataset
    parsed_line = tf.decode_csv(line, record_defaults=[[''], ['']], field_delim='\t')
    parsed_line = dict(zip(HEADER, parsed_line))

    # get label txt column
    label_txt = parsed_line.pop(LABEL)

    return parsed_line, label_txt


def process_seq_padding(sequence):
    """
    Convert token sequence to padded id sequences.
    :param sequence: input text string
    :return: converted word id sequence tensor
    """
    # TODO, if the input sequence are raw word text, you can use tf.contrib.lookup.index_table_from_file to convert
    # word text to word ids, just like bellow.
    # vocab_table = tf.contrib.lookup.index_tabel_from_file(
    #     vocabulary_file=DIR + "vocab_list.txt",
    #     num_oov_buckets=0,
    #     default_value=-1)

    sequence = sequence['text']
    # split text to words -> this will produce sparse tensor with variable lengthes (word counts) entries
    words = tf.string_split(sequence)

    # convert sparse tensor to dense tensor by padding each entry to math the longest in the batch
    dense_words = tf.sparse.to_dense(words, default_value='0')
    dense_words_int = tf.string_to_number(dense_words, tf.int32)

    # convert word to word ids via vocab lookup table
    # word_ids = vocab_table.lookup(dense_words)

    # create word ids padding
    padding = tf.constant([[0,0], [0, MAX_DOC_LEN]], dtype=tf.int32)

    # pad all the word ids entries to the maximum document length
    word_ids_padded = tf.pad(dense_words_int, padding)
    word_id_vector = tf.slice(word_ids_padded, [0,0], [-1, MAX_DOC_LEN])

    # convert data type as your model need
    tf.cast(word_id_vector, tf.int32)

    return word_id_vector


def label_to_ids(label_string_tensor):
    """
    Convert label string to label in
    :param label_string_tensor: label text tensor
    :return:
    """
    table = tf.contrib.lookup.index_table_from_tensor(
        mapping=tf.constant(LABELS), num_oov_buckets=1, default_value=-1)
    return table.lookup(label_string_tensor)


def input_fn(files_name_pattern, mode="train",
             skip_header_lines=0,
             num_epochs=10,
             batch_size=64):
    """
    Create batch input for model, here we assume the input file was csv format
    :param files_name_pattern: file name pattern
    :param mode: to indicate model in train or test
    :param skip_header_lines: skip header
    :param num_epochs: number epochs
    :param batch_size: batch size
    :return: dataset iterator
    """
    shuffle = True if mode == "train" else False
    num_threads = multiprocessing.cpu_count()

    # dataset should be full shuffer because which may influence the final performace of model
    buffer_size = 2 * batch_size + 1

    print("Input file(s): {}".format(files_name_pattern))
    print("Batch size: {}".format(batch_size))
    print("Epoch Count: {}".format(num_epochs))
    print("Mode: {}".format(mode))
    print("Thread Count: {}".format(num_threads))
    print("Shuffle: {}".format(shuffle))

    # file_names = tf.matching_files(files_name_pattern)
    # print("file_names==>", file_names)
    # if dataset was large, can change api to tf.data.TFRecordData from multi tfRecord files
    dataset = tf.data.TextLineDataset(filenames=files_name_pattern)

    dataset = dataset.skip(skip_header_lines)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.map(lambda tsv_row: decode_csv(tsv_row), num_parallel_calls=num_threads)
    dataset = dataset.batch(batch_size)
    # dataset = dataset.padded_batch(batch_size, padded_shapes=[None])
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(buffer_size)

    iterator = dataset.make_initializable_iterator()

    return iterator

files = DIR + "/data.train"
iterator = input_fn(files, num_epochs=2, batch_size=50)

x_txt, y_txt = iterator.get_next()

# parse feature and padding
# TODO, after this process, you can feed x and labels in your models directly , x and y defined in your model do
# TODO, not need define with tf.placehold, you just need define a normal variable.
input_x = process_seq_padding(x_txt)
input_y = label_to_ids(y_txt)


with tf.Session() as sess:
    tf.tables_initializer().run()
    sess.run(iterator.initializer)
    # for i in range(2):
    #     x, y = sess.run([input_x, input_y])
    #     print("input_x==>", x)
    #     print("input_y==>", y)
    i = 0
    while True:
        try:
            x, y = sess.run([input_x, input_y])
            print("============Step:{}==============".format(i))
            print("input_x==>", x)
            print("input_y==>", y)
            i += 1
        except tf.errors.OutOfRangeError:
            print("Complete consume data from tf.dataset!")
            break
