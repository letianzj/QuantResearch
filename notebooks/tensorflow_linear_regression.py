#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date

sample_size = 500
sigma_e = 3.0             # true value of parameter error sigma
random_num_generator = np.random.RandomState(0)
x = 10.0 * random_num_generator.rand(sample_size)
e = random_num_generator.normal(0, sigma_e, sample_size)
y = 1.0 + 2.0 * x +  e          # a = 1.0; b = 2.0; y = a + b*x

import tensorflow as tf

tf.reset_default_graph()            # reset graph
x_plus_i = np.c_[np.ones((500,1)), x]
X = tf.constant(x_plus_i, dtype=tf.float64, name='X')
# keep y as our original dataset; make a copy instead
y_copy = tf.constant(y.reshape(-1,1), dtype=tf.float64, name='y')
X_T = tf.transpose(X)
# Normal equation
beta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(X_T, X)), X_T), y_copy)

with tf.Session() as sess:
    beta_value = sess.run(beta)
    print('pamameters: %.7f, %.7f' % (beta_value[0], beta_value[1]))

import tensorflow as tf

tf.reset_default_graph()
batch_size = 50
n_batches = int(500/batch_size)
n_epochs = 1000
learing_rate = 0.01

# define the graph
w = tf.Variable(tf.truncated_normal([1], mean=0.0, stddev=1.0, dtype=tf.float64, name='slope'))
b = tf.Variable(tf.zeros(1, dtype=tf.float64), name='intercept')
x_ph = tf.placeholder(tf.float64, shape=(None, 1), name='x')
y_ph = tf.placeholder(tf.float64, shape=(None, 1), name='y')
y_pred = tf.add(b, tf.multiply(w, x_ph), name='prediction')
error = y_pred - y_ph
mse = tf.reduce_mean(tf.square(error), name='mse')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learing_rate)
training_op = optimizer.minimize(mse)

# record to TensorBoard
from datetime import datetime
now = datetime.now().strftime('%Y%m%d%H%M%S')
logdir = 'tf_logs/run-{}'.format(now)
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            # Mini-batch gradient descent
            x_batch = x[batch_index*batch_size:(batch_index+1)*batch_size].reshape(-1,1)
            y_batch = y[batch_index * batch_size:(batch_index + 1) * batch_size].reshape(-1, 1)
            sess.run(training_op, feed_dict={x_ph: x_batch, y_ph: y_batch})

            summary_str = mse_summary.eval(feed_dict={x_ph: x_batch, y_ph: y_batch})
            step = epoch *  n_batches + batch_index
            file_writer.add_summary(summary_str, step)

        w_val, b_val = sess.run([w, b])
        print('epoch {}: slope {}, intercept {}'.format(epoch, w_val[0], b_val[0]))

file_writer.close()

