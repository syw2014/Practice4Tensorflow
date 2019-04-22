#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : inference.py
# PythonVersion: python3.6
# Date    : 2019/4/22 下午7:33
# IDE     : PyCharm

"""Inference with trained model"""

import tensorflow as tf
from util_tools import Params, set_logger


import pickle
import argparse
import numpy as np
import os

from model_fn import build_model_spec

# from tensorflow.python import pywrap_tensorflow

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='/root/data/research/data/product/feeds/dataset/output/base_model',
                    help='Directory containing params.json')
parser.add_argument('--data_dir', default='/root/data/research/data/product/feeds/dataset/',
                    help='Directory containing dataset')
parser.add_argument('--restore_dir', default='last_weights',
                    help='Optional, directory containing weights to reload before training')


def create_table(vocab_file):
    """
    Create word lookup table can be instead with tf.lookup table api
    :param vocab_file: string, vocabulary file
    :return: dict, vocabulary with it's integer id
    """
    vocab = open(vocab_file).readlines()
    vocab = [line.strip() for line in vocab]
    word_table = dict(zip(vocab, range(len(vocab))))
    return word_table


def padding(sentence, word_table, params):
    """
    Padding function for single input
    :param sentence: string, input string
    :param word_table: dict, {'word': id}
    :param params: (Object), Paramters
    :return: padded numpy array
    """
    # TODO, other tokenize methods replace here
    tokens = sentence.split(' ')
    token_ids = [word_table[w] for w in tokens if w in word_table]
    if len(token_ids) > params.max_doc_len:
        return token_ids[:params.max_doc_len]   # truncation
    else:
        return token_ids + [0] * (params.max_doc_len - len(token_ids))  # zero padding


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
    assert os.path.isfile(json_path), "No <dataset_params.json> json configuration file found at {}".format(
        args.data_dir)
    params.update(json_path)
    num_oov_buckets = params.num_oov_buckets  # number of buckets for unknown words

    # Set logger
    set_logger(os.path.join(args.model_dir, 'train.log'))
    # print parameters
    params.print()

    # get vocabulary and label filename
    vocab_path = os.path.join(args.data_dir, 'words.txt')
    label_path = os.path.join(args.data_dir, 'tags.txt')

    # Create word lookup table and label lookup table
    # words_table = tf.contrib.lookup.index_table_from_file(vocab_path, num_oov_buckets=num_oov_buckets)
    tag_id_table = tf.contrib.lookup.index_to_string_table_from_file(
        vocabulary_file=label_path, default_value="UNKNOW")

    vocab_dict = create_table(vocab_path)

    # reader = pywrap_tensorflow.NewCheckpointReader(args.model_dir + '/mymodel')
    # var_to_shape_map = reader.get_variable_to_shape_map()
    # for key in var_to_shape_map:
    #     print('tensor_name: ', key)

    with open(args.model_dir + '/mymodel.pkl', 'rb') as fin:
        model = pickle.load(fin)

    # define placeholder for inference
    input_sentence = tf.placeholder(tf.int32, shape=[1, params.max_doc_len])
    model_spec = build_model_spec('infer', model, {'sentence': input_sentence}, params)
    predictions = model_spec['prediction']
    scores = model_spec['score']
    with tf.Session() as sess:
        saver = tf.train.Saver()
        save_path = os.path.join(args.model_dir, args.restore_dir)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        print("save_path: ", save_path)
        # tf.train.Saver().restore(sess, save_path)
        saver.restore(sess, save_path)
        test_sents = ["你 给 的 杯 子 尺 寸 不 对 啊 智 障 ！",
                      "【 贺 兰 大 人 】   第 三 集 c u t",
                      "爱 神 巧 克 力   ， 浩 一 的 妹 妹 真 会 开 玩 笑",
                      " 手 書 き 】 来 神 高 校 生 の 日 常 【 D R R R 】",
                      "我 们 童 年 时 的 游 戏 大 红 人 ， 不 知 大 家 还 记 得 不",
                      "男 子 刚 准 备 干 活 ， 突 然 感 觉 不 对 劲 ， 监 控 拍 下 可 怕 一 幕 ！",
                      "《 加 勒 比 海 盗 ： 黑 珍 珠 号 的 诅 咒 》   预 告 片",
                      "前 边 骑 电 动 车 的 护 士 姐 姐 ， 裙 子 穿 这 么 短 ， 就 不 担 心 后 面 撞 车 吗",
                      "超 游 世 界 日 语 版   1 4   我 的 同 级 生 和 青 梅 竹 马 的 惨 烈 修 罗 场",
                      "人 民 币 的 历 史 ， 说 的 好 有 道 理",
                      "欢 迎 来 到 夏 天 G o d 的 直 播 间"]
        sess.run(tf.tables_initializer())

        # create input
        for txt in test_sents:
            # input = np.random.randint(0, 100000, size=[150])
            input = padding(txt, vocab_dict, params)
            label_txt = tag_id_table.lookup(predictions)

            pred, classes, score = sess.run([predictions, label_txt, scores], feed_dict={input_sentence: [input]})
            classes = [tf.compat.as_str_any(x) for x in classes]
            score = [np.max(x) for x in score]
            # max_node = np.argmax(pred)
            print("predicted label index: ", pred, classes, score)


if __name__ == "__main__":
    main()
