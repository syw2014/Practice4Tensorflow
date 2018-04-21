#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# File: tensorpack_test.py
# Date: 18-4-11 上午11:03

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

import json

# writer1 = tf.python_io.TFRecordWriter("../../data/tmp/test1.tfrecord")
# writer2 = tf.python_io.TFRecordWriter("../../data/tmp/test2.tfrecord")
#
#
# def _bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
#
#
# def _bytes_feature_list(values):
#     return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])
#
#
# with open("../../data/test.json") as f:
#     for idx, line in enumerate(f.readlines()):
#         sample = json.loads(line)
#
#         # tmp = sample
#         if isinstance(sample, str):
#             tmp = sample
#         elif isinstance(sample, unicode):
#             tmp = sample.encode('utf8', 'ignore')
#         else:
#             tmp = str(sample)
#
#         if idx == 0:
#             writer = writer1
#         else:
#             writer = writer2
#         # example = tf.train.Example(features=tf.train.Features(feature={
#         #     "sample": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tmp]))
#         # }))
#         feature_lists = tf.train.FeatureLists(
#             feature_list={
#                 "sample": _bytes_feature_list([tmp])
#             })
#         sequence_example = tf.train.SequenceExample(feature_lists=feature_lists)
#
#         writer.write(sequence_example.SerializeToString())
#
# writer1.close()
# writer2.close()

# parse tfrecord
tfrecords = "../../data/tmp/train-00001-of-00002"
filename_qu = tf.train.string_input_producer([tfrecords])

reader = tf.TFRecordReader()
_, se_example = reader.read(filename_qu)

features = tf.parse_single_sequence_example(
    se_example,
    sequence_features={
        "sample": tf.FixedLenSequenceFeature([], tf.string)
    }
)

sstr = features[1]
print(features)
# sstr = tf.cast(features['sample'], tf.string)

init = tf.initialize_all_variables()

with tf.Session() as sess:

    # sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(3):
        res = sess.run([sstr])
        # l = to_categorical(l, 12)
        print(res)

