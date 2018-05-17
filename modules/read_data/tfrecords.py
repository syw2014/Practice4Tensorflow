#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# File: tfrecords.py
# Date: 18-4-21 上午11:19


"""This example is different as others , here I use the text data to show how to create tfrecords and use tensorflow
queue to read tfrecords.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime

import tensorflow as tf
import numpy as np
import threading
import argparse
import random
import sys
import os


# parser = argparse.ArgumentParser()
# parser.add_argument("--output_dir", help="The output directory of TFRecords", type=str)
# args = parser.parse_args()

output_dir = "../../data/tmp/"


def _bytes_feature(value):
    """Wrapper for inserting a bytes or string feature to Example  proto."""

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _to_sequence_example(data):
    """Builds a SequenceExample proto for your input data.
    Args:
        data: The data is your input, and you should make your own parser. As an example here I use a text json to show
                how to build sequence example.
    Returns:
        A SequenceExample proto.
    """
    # As in this example there is no context
    # context = tf.train.Feature()
    # TODO, more process for your data
    feature_lists = tf.train.FeatureLists(feature_list={
        "sample": _bytes_feature_list([data])
    })

    sequence_example = tf.train.SequenceExample(
        # Here , context is none, you can define your own context
        feature_lists=feature_lists
    )

    return sequence_example


def _process_one_files(thread_index, ranges, name, meta_data, num_shards):
    """Processes andd saves a subset of meta data as TFRecord files in one threadd.
    Args:
        thread_index: Integer thread identifier within [0, len(range)]
        ranges:   A list of pairs of integers specifying the ranges of the dataset to
        process in parallel
        name: Unique identifier specifying the dataset
        meta_data: List of raw data
        num_shards: Integer number of shards for the output files
    """
    # Each thread produces N shards where N = num_shards / num_threads. For
    # instance, if num_shards = 128 and num_threads = 2, then the first thread
    # would produce shards [0, 64)
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_data_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a shards version of the file name, eg. 'train-00001-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
        output_file = os.path.join(output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        data_in_shard = np.arange(shard_ranges[s], shard_ranges[s+1], dtype=int)
        for i in data_in_shard:
            data = meta_data[i]

            sequeue_example = _to_sequence_example(data)
            if sequeue_example is not None:
                writer.write(sequeue_example.SerializeToString())
                shard_counter += 1
                counter += 1

            # print infos
            if not counter % 1000:
                print("%s [thread %d]: Processed %d of %d items in thread batch" %
                      (datetime.now(), thread_index, counter, num_data_in_thread))
                sys.stdout.flush()

        writer.close()
        print("%s [thread %d]: Wrote %d data to %s" %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()

    print("%s [thread %d]: Wrote %d data to %d shards" %
          (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()


def _process_dataset(name, dataset, num_shards, set_threads):
    """Process a complete dataset and saves as a TFRecord.
    Args:
        name: Unique identifier specifying the dataset
        dataset: List of input data
        num_shards: Integer number of shards for the output files.
        num_threads: Integer number of thread
    """
    # Shuffle the ordering of images, Make the randomization repeatable
    random.seed(1234)
    random.shuffle(dataset)

    #
    num_threads = min(num_shards, set_threads)
    spacing = np.linspace(0, len(dataset), num_threads+1).astype(np.int)
    ranges = []
    threads = []
    for i in range(len(spacing)-1):
        ranges.append([spacing[i], spacing[i+1]])

    # Create a mechanism for monitoring when all threads are finished
    coord = tf.train.Coordinator()

    # Launch a thread for each batch.
    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, dataset,  num_shards)
        t = threading.Thread(target=_process_one_files, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d data pairs in data set '%s'." %
          (datetime.now(), len(dataset), name))


def parse_sequence_example(serialized, sample_feature):
    """Parse a tensorflow.SequenceExample into an real sample.
    Args:
        serialized: A scalar string Tensor, a single serialized SequenceExample.
        sample_feature: Name of SequenceExample feature list you have set in Serialized
    Return:
        A raw sample.
    """
    _, sequence = tf.parse_single_sequence_example(
        serialized,
        # Here I have not context when convert to sequence example, context features is none
        # context_features= None
        sequence_features={
            sample_feature: tf.FixedLenSequenceFeature([], dtype=tf.string)
        })

    sample = sequence['sample']

    return sample


def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
    """Prefetches string values from disk into an input queue.

    In training the capacity of the queue is important because a larger queue
    means better mixing of training examples between shards. The minimum number of
    values kept in the queue is values_per_shard * input_queue_capacity_factor,
    where input_queue_memory factor should be chosen to trade-off better mixing
    with memory usage.

    Args:
        reader: Instance of tf.ReaderBase.
        file_pattern: Comma-separated list of file patterns (e.g.
            /tmp/train_data-?????-of-00100).
        is_training: Boolean; whether prefetching for training or eval.
        batch_size: Model batch size used to determine queue capacity.
        values_per_shard: Approximate number of values per shard.
        input_queue_capacity_factor: Minimum number of values to keep in the queue
        in multiples of values_per_shard. See comments above.
        num_reader_threads: Number of reader threads to fill the queue.
        shard_queue_name: Name for the shards filename queue.
        value_queue_name: Name for the values input queue.

    Returns:
        A Queue containing prefetched string values.
  """
    data_files = []
    for pattern in file_pattern.split(","):
        data_files.extend(tf.gfile.Glob(pattern))
    if not data_files:
        tf.logging.fatal("Found no input files matching %s", file_pattern)
    else:
        tf.logging.info("Prefetching values from %d files matching %s",
                        len(data_files), file_pattern)

    # If in training model data will be shuffled, otherwise is first in first out
    if is_training:
        filename_queue = tf.train.string_input_producer(
            data_files, shuffle=True, capacity=16, name=shard_queue_name)
        min_queue_examples = values_per_shard * input_queue_capacity_factor
        capacity = min_queue_examples + 100 * batch_size
        values_queue = tf.RandomShuffleQueue(
            capacity=capacity,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string],
            name="random_" + value_queue_name)
    else:
        filename_queue = tf.train.string_input_producer(
            data_files, shuffle=False, capacity=1, name=shard_queue_name)
        capacity = values_per_shard + 3 * batch_size
        values_queue = tf.FIFOQueue(
            capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

    # Create reader queue
    enqueue_ops = []
    for _ in range(num_reader_threads):
        _, value = reader.read(filename_queue)
        enqueue_ops.append(values_queue.enqueue([value]))
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
          values_queue, enqueue_ops))
    tf.summary.scalar(
          "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
        tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

    return values_queue


def get_data(
        reader,
        file_pattern,
        is_training,
        batch_size,
        num_preprocess_threads,
        values_per_shard,
        input_queue_capacity_factor=16,
        num_reader_threads=1):
    """Get raw sample from prefetch mechanism. This was im
    Args:
        reader: Instance of tf.ReaderBase.
        file_pattern: Comma-separated list of file patterns (e.g.
            /tmp/train_data-?????-of-00100).
        is_training: Boolean; whether prefetching for training or eval.
        batch_size: Model batch size used to determine queue capacity.
        num_preprocess_threads: Integer number of preprocess threads
        values_per_shard: Approximate number of values per shard.
        input_queue_capacity_factor: Minimum number of values to keep in the queue
        in multiples of values_per_shard. See comments above.
        num_reader_threads: Number of reader threads to fill the queue.

    """
    input_queue = prefetch_input_data(
        reader,
        file_pattern,
        is_training,
        batch_size,
        values_per_shard=values_per_shard,
        input_queue_capacity_factor=input_queue_capacity_factor,
        num_reader_threads=num_reader_threads)
    # Split across multiple threads with each thread applying a slightly different distortion.
    assert num_preprocess_threads % 2 == 0
    samples = []
    for thread_id in range(num_preprocess_threads):
        serialized_sequence_example = input_queue.dequeue()
        sample = parse_sequence_example(serialized_sequence_example, "sample")
        samples.append(sample)

    return samples


def main():
    # print("Test TFRecods generation")
    # with open("../../data/test.json") as f:
    #     dataset = f.readlines()
    #     _process_dataset("train", dataset, 2, 2)

    # Parse TFRecords
    reader = tf.TFRecordReader()
    init = tf.initialize_all_variables()
    samples = get_data(
        reader,
        "/data/research/github/repos/Practice4Tensorflow/data/tmp/train-?????-of-00002",
        True,
        12,
        4,
        2300)

    with tf.Session() as sess:
        sess.run(init)
        for x in samples:
            print(tf.cast(x, tf.string))


if __name__ == "__main__":
    main()