#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# -------------------------------------------- Data ------------------------------------------------------- #
# price to return normalization
spx = pd.read_csv('hist/SPX Index.csv', index_col=0, header=0)
spx = spx[['High', 'Low', 'Close']]
spx_close = spx[['Close']*3]
spx_close.columns = spx.columns
spx_ret = (spx / spx_close.shift(1) - 1)*100.0      # multiply by 100; in percentage
spx_ret.dropna(inplace=True)            # shape = (3059, 3), from 1/4/2006 to 2/28/2018

n_window_size = 20            # 20 business days; use first 19 to predict the 20th
# split between train and test, 90%/10%
n_total_size = spx_ret.shape[0] - n_window_size + 1             # 3040
n_train_set_size = int(np.round(n_total_size*0.9))          # 2736
n_test_set_size = n_total_size - n_train_set_size           # 304

x_train = np.zeros((n_train_set_size, n_window_size-1, 3))          # shape = (2736, 19, 3)
y_train = np.zeros((n_train_set_size, 1))                           # shape = (2736, 1)
x_test = np.zeros((n_test_set_size, n_window_size-1, 3))            # shape = (304, 19, 3)
y_test = np.zeros((n_test_set_size, 1))                             # shape = (304, 1)

for i in range(n_train_set_size):
    x_train[i, :, :] = spx_ret.iloc[i:i+n_window_size-1].values
    y_train[i, 0] = spx_ret.iloc[i + n_window_size - 1, 2]

for i in range(n_train_set_size, n_total_size):
    x_test[i-n_train_set_size, :, :] = spx_ret.iloc[i:i+n_window_size-1].values
    y_test[i-n_train_set_size, 0] = spx_ret.iloc[i + n_window_size - 1, 2]

# generate next batch; randomly shuffle test set; and then draw without replacement batch_size samples
# after running out of samples, randomly shuffle again
index_in_epoch = 0
perm_array = np.arange(x_train.shape[0])            # (2736,)
np.random.shuffle(perm_array)


# function to get the next batch; randomly draw batch_size(50) 20d-windows
def get_next_batch(batch_size):
    global index_in_epoch, x_train, perm_array
    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array)  # shuffle permutation array
        start = 0  # start next epoch
        index_in_epoch = batch_size

    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]


# -------------------------------------------- Build Graph ------------------------------------------------------- #
n_steps = n_window_size - 1                 # 20 business days, X has 19 days
n_inputs = 3                                # HLC
n_outputs = 1                               # C(t=20)
n_neurons = 200                             # number of neurons in a layer
n_layers = 2                                # number of layers
learning_rate = 0.001                       # learning rate
batch_size = 50                             # batch size
n_epochs = 100                              # number of epochs

tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])                   # (?, 19, 3)
y = tf.placeholder(tf.float32, [None, n_outputs])                           # (?, 1)

# Bonus: Multi-Layer Perceptron (MLP)
# X_MLP = tf.placeholder(tf.float32, [None, n_steps * n_inputs])              # (?, 57)
# hidden1 = tf.contrib.layers.fully_connected(X_MLP, n_neurons, activation_fn=tf.nn.elu)      # (?, 200)
# hidden2 = tf.contrib.layers.fully_connected(hidden1, n_neurons, activation_fn=tf.nn.elu)      # (?, 200)
# y_pred_MLP = tf.contrib.layers.fully_connected(hidden2, n_outputs, activation_fn=tf.nn.elu)     # (?, 1)

# Basic RNN Cell
layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu) for layer in range(n_layers)]

# Basic LSTM Cell
# layers = [tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', num_units=n_neurons, activation=tf.nn.elu) for layer in range(n_layers)]

# LSTM Cell with peephole connections
# layers = [tf.contrib.rnn.LSTMCell(num_units=n_neurons, activation=tf.nn.leaky_relu, use_peepholes = True) for layer in range(n_layers)]

# GRU cell
# layers = [tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.leaky_relu) for layer in range(n_layers)]

multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)

# rnn_outputs contains the output tensors for each time step (?, 19, 200)
# states contains the final states of the network, (?, 200)x(2 layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

# add a densely-connected layer between final state and y
y_pred = tf.layers.dense(states[-1], n_outputs)                 # (?, 1)

loss = tf.reduce_mean(tf.square(y_pred - y))          # loss function is mean squared error
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
model_saver = tf.train.Saver()

# -------------------------------------------- Run Graph ------------------------------------------------------- #
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        for iteration in range(n_train_set_size // batch_size):
            x_batch, y_batch = get_next_batch(batch_size) # fetch the next training batch
            #[a1, a2, a3, a4] = sess.run([rnn_outputs, states, y_pred, training_op], feed_dict={X: x_batch, y: y_batch})
            sess.run(training_op, feed_dict={X: x_batch, y: y_batch})

        mse_train = loss.eval(feed_dict={X: x_train, y: y_train})
        mse_test = loss.eval(feed_dict={X: x_test, y: y_test})
        print(epoch, "Train accuracy:", mse_train, "Test accuracy:", mse_test)

    y_train_pred = sess.run(y_pred, feed_dict={X: x_train})                 # (2736, 1)
    y_test_pred = sess.run(y_pred, feed_dict={X: x_test})                   # (304, 1)
    save_path = model_saver.save(sess, "./rnn_model_final.ckpt")        # checkpoint

# -------------------------------------------- Plot Results ------------------------------------------------------- #
# transform from return back to price
y_train_actual = spx['Close'].iloc[n_window_size:n_train_set_size+n_window_size]             # (2736,)
y_test_actual = spx['Close'].iloc[n_train_set_size+n_window_size:]                           # (304, )

y_train_pred_price = pd.DataFrame(y_train_pred, index=y_train_actual.index)
y_train_pred_price.columns = ['Close']
y_train_pred_price = spx[['Close']].shift(1).iloc[n_window_size:n_train_set_size+n_window_size] * (y_train_pred_price/100.0 + 1.0)
y_test_pred_price = pd.DataFrame(y_test_pred, index=y_test_actual.index)
y_test_pred_price.columns = ['Close']
y_test_pred_price = spx[['Close']].shift(1).iloc[n_train_set_size+n_window_size:] * (y_test_pred_price/100.0 + 1.0)

# plot
plt.plot(y_train_actual.index, y_train_actual, color='blue', label='train actual')
plt.plot(y_train_pred_price.index, y_train_pred_price, color='red', label='train prediction')
plt.plot(y_test_actual.index, y_test_actual, color='yellow', label='test actual')
plt.plot(y_test_pred_price.index, y_test_pred_price, color='green', label='test prediction')
plt.show()

