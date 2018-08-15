#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# File: project_template.py
# Date: 18-5-17 上午9:11

"""Here is a simple template of deep learning project with tensorflow, according to this you can design your onw project
by modify some modules. This is the basic use of tensorflow, future we will make more high level use with tf estimator."""

import tensorflow as tf
import numpy as np

# [1] define hyperparameters
learning_rate = 0.01
max_train_steps = 10000


# [2] prepare train data
train_x = np.array([[x] for x in np.arange(1, 17, 1)], dtype=np.float32)
train_y = np.array([[y] for y in [4.00, 6.40, 8.00, 8.80, 9.22, 9.50, 9.70, 9.86, 10.00, 10.20,
                    10.32, 10.42, 10.50, 10.55, 10.58, 10.60]], dtype=np.float32)

num_sample = train_x.shape[0]
print(num_sample)
# [3] construct model
# input data
X = tf.placeholder(tf.float32, [None, 1])
y_ = tf.placeholder(tf.float32, [None, 1])

# weights
W = tf.Variable(tf.random_normal([1, 1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

# construct model
Y = tf.matmul(X, W) + b

# [4] define loss
# MSE
loss = tf.reduce_mean(tf.pow(Y-y_, 2)) / num_sample

# [5] create optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)


# [6] define single train step
train_op = optimizer.minimize(loss)


# [7] create tensorflow session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    # initialize global variables
    sess.run(tf.global_variables_initializer())

    # [8] train model
    print("Start training...")
    for step in xrange(max_train_steps):
        sess.run(train_op, feed_dict={X: train_x, y_: train_y})

        # print logs
        if step % 10 == 0:
            cost = sess.run(loss, feed_dict={X: train_x, y_: train_y})
            print("Step: {}, loss={}, W={}, b={}".format(step, cost, sess.run(W), sess.run(b)))

    # final loss
    final_loss = sess.run(loss, feed_dict={X: train_x, y_: train_y})
    print("Step: {}, loss={}, W={}, b={}".format(max_train_steps, final_loss, sess.run(W), sess.run(b)))
    weight, bias = sess.run([W, b])
    print("Linear Regression Model: Y={}*X + {}".format(weight[0][0], bias[0]))