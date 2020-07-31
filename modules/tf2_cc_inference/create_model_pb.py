#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : create_model_pb.py
# PythonVersion: python3.6
# Date    : 2020/7/30 15:44
# Software: PyCharm
"""Saved model to pb format with TF2.x.
Here we create a simple model with TF2.x, then load model and inference with TF2.x C++ api"""


import tensorflow as tf
import numpy as np


class TestModel(tf.keras.Model):
    def __init__(self):
        super(TestModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(1, kernel_initializer='Ones', activation=tf.nn.relu)
    
    def call(self, inputs):
        return self.dense1(inputs)


if __name__ == "__main__":
    input_data = np.asarray([[10]])
    model = TestModel()
    # But in order to save the model ( refer to this line module.save('model', save_format='tf')), 
    # the graph needs to be built before it can save. hence we will need to call the model at 
    # least once for it to create the graph.
    # Calling print(module(input_data)) will force it to create the graph
    # You also can train the model before save instead of feed input
    model._set_inputs(input_data)
    print(input_data.shape)
    output = model(input_data)
    print(output)

    # save model
    tf.saved_model.save(model, "./result/pb")
