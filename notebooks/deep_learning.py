#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# fetch MNIST data
from sklearn.datasets import fetch_mldata
try:
    mnist = fetch_mldata('MNIST original')
except Exception as ex:
    from six.moves import urllib
    from scipy.io import loadmat
    import os

    mnist_path = os.path.join(".", "datasets", "mnist-original.mat")

    # download dataset from github.
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    response = urllib.request.urlopen(mnist_alternative_url)
    with open(mnist_path, "wb") as f:
        content = response.read()
        f.write(content)

    mnist_raw = loadmat(mnist_path)
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }
    print("Done!")


X, y = mnist['data'], mnist['target']
print(X.shape, y.shape)

import matplotlib
import matplotlib.pyplot as plt

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
plt.axis('off')
plt.show()

# ---------------------------- Plain-vanilla two hidden-layer feed-forward ----------------------- #
# This is an alternative to the tf.fully_connected
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X, W) + b
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z

import tensorflow as tf
n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None), name='y')

from tensorflow.contrib.layers import fully_connected
with tf.name_scope('dnn'):
    hidden1 = fully_connected(X, n_hidden1, scope='hidden1')
    hidden2 = fully_connected(hidden1, n_hidden2, scope='hidden2')
    logits = fully_connected(hidden2, n_outputs, scope='outputs', activation_fn=None)

with tf.name_scope('loss'):
    # equivalent to applying the softmax activation function and then computing the cross entropy
    # softmax transforms outputs into probabilities;
    # logistic function (binary) turns one dimensional scalar into probability; vs softmax handles 10 dimensions
    # cross entropy gives errors similar to MLE
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

# optimize
learning_rate = 0.01
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    # Says whether the targets are in the top K predictions; returns boolean
    correct = tf.nn.in_top_k(logits, y, 1)
    # cast boolean into float and then take average
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
accuracy_summary = tf.summary.scalar('accuracy', accuracy)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data')

n_epochs = 30       # 400
batch_size = 50

# Execution
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)           # mini-batch
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        # acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})       # using last batch
        acc_train, summary_str = sess.run([accuracy, accuracy_summary], feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        file_writer.add_summary(summary_str, epoch)
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./my_model_final.ckpt")           # checkpoint
    file_writer.close()

# check tensorboard
# tensorboard --logdir tf_logs/

# Restore and Use
import numpy as np
with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    X_new_scaled = mnist.test.images[0].reshape(1, -1)
    Z = logits.eval(feed_dict={X: X_new_scaled})
    print(tf.nn.softmax(Z).eval())            # output all the estimated class probabilities
    y_pred = np.argmax(Z, axis=1)    # just want to know the class with highest logit value
    print(mnist.test.labels[0], y_pred)

# ----------------------- End of Plain-vanilla two hidden-layer feed-forward ----------------------- #

# -------------------------------- Basic RNN  ------------------------------------------------------ #
n_inputs = 3            # 3d tensor, e.g., (open, close, volume)
n_neurons = 5
tf.reset_default_graph()

X0 = tf.placeholder(tf.float32, [None, n_inputs])       # None = batch size
X1 = tf.placeholder(tf.float32, [None, n_inputs])

Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
b = tf.Variable(tf.zeros(shape=[1, n_neurons], dtype=tf.float32))

Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(X1, Wx) + tf.matmul(Y0, Wy) + b)

init = tf.global_variables_initializer()

# Mini-batch: instance 0,instance 1,instance 2,instance 3
# shape = 4x3, row=batch, col=input
X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})

# --------------------------  End of Basic RNN --------------------------------------------------- #

# -------------------------------- RNN static Unrolling ------------------------------------------------- #
n_inputs = 3
n_neurons = 5
tf.reset_default_graph()

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)
Y0, Y1 = output_seqs

init = tf.global_variables_initializer()

# Mini-batch: instance 0,instance 1,instance 2,instance 3
# shape = 4x3, row=batch, col=input
X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})

# another way
tf.reset_default_graph()
n_steps = 2

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2])) # unstack into two steps
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)
outputs = tf.transpose(tf.stack(output_seqs), perm=[1, 0, 2])

X_batch = np.array([
    # t = 0 t = 1
    [[0, 1, 2], [9, 8, 7]], # instance 0
    [[3, 4, 5], [0, 0, 0]], # instance 1
    [[6, 7, 8], [6, 5, 4]], # instance 2
    [[9, 0, 1], [3, 2, 1]], # instance 3
])

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X: X_batch})

# -------------------------------- End of RNN static Unrolling ---------------------------------------------- #

# -------------------------------- RNN dynamic Unrolling ------------------------------------------------- #
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

# ------------------------------- End of RNN dynamic Unrolling ------------------------------------------- #

# ------------------------------------------ RNN MNIST -------------------------------------------------- #
n_steps = 28
n_inputs = 28       # number of X in each step
n_neurons = 150
n_outputs = 10
tf.reset_default_graph()

learning_rate = 0.001
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
logits = fully_connected(states, n_outputs, activation_fn=None)          # connect end state to ten-state logit
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)      # softmax plus cross entropy
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

# load data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels

n_epochs = 50
batch_size = 150
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

# --------------------------------------- End of RNN MNIST -------------------------------------------------- #

# -------------------------------------- RNN time series -------------------------------------------------- #
import tensorflow as tf
n_steps = 20
n_inputs = 1     # one feature
n_neurons = 100
n_outputs = 1
tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# project from 100 ==> 1, FC (fully connected)
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
    output_size=n_outputs)

learning_rate = 0.001
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()

n_iterations = 10000
batch_size = 50
with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = [...]  # fetch the next training batch
    sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    if iteration % 100 == 0:
        mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
    print(iteration, "\tMSE:", mse)

# make prediction
X_new = [...] # New sequences
y_pred = sess.run(outputs, feed_dict={X: X_new})

# Creative RNN: add one-step prediction back into X, then predict one more step
sequence = [0.] * n_steps
for iteration in range(300):
    X_batch = np.array(sequence[-n_steps:]).reshape(1, n_steps, 1)
    y_pred = sess.run(outputs, feed_dict={X: X_batch})
    sequence.append(y_pred[0, -1, 0])


# Deep RNN
n_neurons = 100
n_layers = 3
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
multi_layer_cell = tf.contrib.rnn.MultiRNNCell([basic_cell] * n_layers)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

# dropout
import sys
is_training = (sys.argv[-1] == "train")
keep_prob = 0.5
cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
if is_training:     # dorpout should only be applied to training, not testing
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
multi_layer_cell = tf.contrib.rnn.MultiRNNCell([cell] * n_layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

with tf.Session() as sess:
    if is_training:
        init.run()
        for iteration in range(n_iterations):
            [...] # train the model
        save_path = saver.save(sess, "/tmp/my_model.ckpt")
    else:
        saver.restore(sess, "/tmp/my_model.ckpt")
        [...] # use the model

# ---------------------------------- End of RNN time series ------------------------------------------------ #

# ------------------------------------------- AutoEncoder ------------------------------------------------- #
# If the autoencoder uses only linear activations and the cost function is the Mean Squared Error (MSE),
# then it can be shown that it ends up performing Principal Component Analysis
# The number of outputs is equal to the number of inputs.
# To perform simple PCA, we set activation_fn=None (i.e., all neurons are linear)
# and the cost function is the MSE.

# stacked AutoEncoder MNIST
n_inputs = 28 * 28 # for MNIST
n_hidden1 = 300
n_hidden2 = 150 # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs       # restore

learning_rate = 0.01
l2_reg = 0.001
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
with tf.contrib.framework.arg_scope(
        [fully_connected],
        activation_fn=tf.nn.elu,            # ELU activation function,
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),      # He initialization
        weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg)):      # L2 regularization
    hidden1 = fully_connected(X, n_hidden1)
    hidden2 = fully_connected(hidden1, n_hidden2) # codings
    hidden3 = fully_connected(hidden2, n_hidden3)
    outputs = fully_connected(hidden3, n_outputs, activation_fn=None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))    # MSE
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_epochs = 5
batch_size = 150
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            # training_op.run(feed_dict={X: X_batch})  # no labels (unsupervised)
            sess.run(training_op, feed_dict={X: X_batch})

    # load X_test and reconstruct
    outputs_val = outputs.eval(feed_dict={X: X_test})

# Tying Weights: tie the weights of the decoder layers to the weights of the encoder layers.
# This halves the number of weights in the model, speeding up training and limiting the risk of overfitting.

# --------------------------------------- End of AutoEncoder ------------------------------------------------ #

# ------------------------------------- Reinforcement Learning --------------------------------------------- #
# Policy Gradient


# --------------------------------- End of Reinforcement Learning --------------------------------------------- #


# ----------------------------------------- Stock LSTM ---------------------------------------------------- #
# https://www.kaggle.com/raoulma/ny-stock-price-prediction-rnn-lstm-gru/notebook
# https://medium.com/@alexrachnog/neural-networks-for-algorithmic-trading-part-one-simple-time-series-forecasting-f992daa1045a
# https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/ <-- for arima, ols
import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf

# split data in 80%/10%/10% train/validation/test sets
valid_set_size_percentage = 10
test_set_size_percentage = 10

# shape = (851,264, 6), symbolOHLCV, daily
df = pd.read_csv("research/prices-split-adjusted.csv", index_col=0)
df.info()
df.head()
df.tail()
df.describe()
df.info()

plt.figure(figsize=(15, 5));
plt.subplot(1,2,1);
plt.plot(df[df.symbol == 'EQIX'].open.values, color='red', label='open')
plt.plot(df[df.symbol == 'EQIX'].close.values, color='green', label='close')
plt.plot(df[df.symbol == 'EQIX'].low.values, color='blue', label='low')
plt.plot(df[df.symbol == 'EQIX'].high.values, color='black', label='high')
plt.title('stock price')
plt.xlabel('time [days]')
plt.ylabel('price')
plt.legend(loc='best')

plt.subplot(1,2,2)
plt.plot(df[df.symbol == 'EQIX'].volume.values, color='black', label='volume')
plt.title('stock volume')
plt.xlabel('time [days]')
plt.ylabel('volume')
plt.legend(loc='best')
plt.show()

# choose a specific stock, normalize price
# function for min-max normalization of stock
def normalize_data(df):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    df['close'] = min_max_scaler.fit_transform(df['close'].values.reshape(-1,1))
    return df


# function to create train, validation, test data given stock data and sequence length
# use previous 19 days to predict today
def load_data(stock, seq_len):
    data_raw = stock.as_matrix()  # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - seq_len):        # 1762 - 20
        data.append(data_raw[index: index + seq_len])

    data = np.array(data)           # (1742, 20, 4)
    valid_set_size = int(np.round(valid_set_size_percentage / 100 * data.shape[0]))         # 174
    test_set_size = int(np.round(test_set_size_percentage / 100 * data.shape[0]))           # 174
    train_set_size = data.shape[0] - (valid_set_size + test_set_size)                       # 1394

    x_train = data[:train_set_size, :-1, :]         # first 19, (1394, 19, 4)
    y_train = data[:train_set_size, -1, :]          # the last one, (1394, 4)

    x_valid = data[train_set_size:train_set_size + valid_set_size, :-1, :]
    y_valid = data[train_set_size:train_set_size + valid_set_size, -1, :]

    x_test = data[train_set_size + valid_set_size:, :-1, :]
    y_test = data[train_set_size + valid_set_size:, -1, :]

    return [x_train, y_train, x_valid, y_valid, x_test, y_test]

# choose one stock
df_stock = df[df.symbol == 'EQIX'].copy()
df_stock.drop(['symbol'],1,inplace=True)
df_stock.drop(['volume'],1,inplace=True)

cols = list(df_stock.columns.values)
print('df_stock.columns.values = ', cols)

# normalize stock
df_stock_norm = df_stock.copy()
df_stock_norm = normalize_data(df_stock_norm)

# create train, test data
seq_len = 20 # choose sequence length
x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(df_stock_norm, seq_len)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_valid.shape = ',x_valid.shape)
print('y_valid.shape = ', y_valid.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ',y_test.shape)

## Basic Cell RNN in tensorflow
index_in_epoch = 0
perm_array = np.arange(x_train.shape[0])            # (1394,)
np.random.shuffle(perm_array)

# function to get the next batch; randomly draw 50 20d-windows
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

# parameters
n_steps = seq_len-1             # 19
n_inputs = 4                    # ohlc
n_neurons = 200
n_outputs = 4                   # ohlc
n_layers = 2
learning_rate = 0.001
batch_size = 50
n_epochs = 100
train_set_size = x_train.shape[0]
test_set_size = x_test.shape[0]

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

# TODO: add MLP

# use Basic RNN Cell
layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu) for layer in range(n_layers)]

# use Basic LSTM Cell
#layers = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.elu)
#          for layer in range(n_layers)]

# use LSTM Cell with peephole connections
#layers = [tf.contrib.rnn.LSTMCell(num_units=n_neurons,
#                                  activation=tf.nn.leaky_relu, use_peepholes = True)
#          for layer in range(n_layers)]

# use GRU cell
#layers = [tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.leaky_relu)
#          for layer in range(n_layers)]

# TODO: dropout wrapper; use name_scope
# http://androidkt.com/stock-price-prediction/
# https://lilianweng.github.io/lil-log/2017/07/08/predict-stock-prices-using-RNN-part-1.html

multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
# rnn_outputs contains the output tensors for each time step (?, 19, 200)
# states contains the final states of the network, (?, 200)x(2 layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

# TODO: states ==> output directly; currently it connects all 19 steps to the output layer
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])      # (?, 19, 200) ==> (?*19, 200)
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)   # (?*19, 200) ==> (?*19, 4)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])     # (?*19,4) ==> (?, 19, 4)
outputs = outputs[:, n_steps-1, :]    # keep only last output of sequence  # (?, 19, 4) ==> (?, 4)

loss = tf.reduce_mean(tf.square(outputs - y))   # loss function = mean squared error
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

# run graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iteration in range(int(n_epochs*train_set_size/batch_size)):
        x_batch, y_batch = get_next_batch(batch_size) # fetch the next training batch
        #sess.run(training_op, feed_dict={X: x_batch, y: y_batch})
        [a1, a2, a3, a4, a5] = sess.run([rnn_outputs, stacked_rnn_outputs, stacked_outputs, outputs, training_op], feed_dict={X: x_batch, y: y_batch})
        if iteration % int(5*train_set_size/batch_size) == 0:
            mse_train = loss.eval(feed_dict={X: x_train, y: y_train})
            mse_valid = loss.eval(feed_dict={X: x_valid, y: y_valid})
            print('%.2f epochs: MSE train/valid = %.6f/%.6f'%(
                iteration*batch_size/train_set_size, mse_train, mse_valid))

    y_train_pred = sess.run(outputs, feed_dict={X: x_train})
    y_valid_pred = sess.run(outputs, feed_dict={X: x_valid})
    y_test_pred = sess.run(outputs, feed_dict={X: x_test})

# prediction
ft = 0      # 0 = open, 1 = close, 2 = highest, 3 = lowest

## show predictions
plt.figure(figsize=(15, 5));
plt.subplot(1,2,1);

plt.plot(np.arange(y_train.shape[0]), y_train[:,ft], color='blue', label='train target')

plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_valid.shape[0]), y_valid[:,ft],
         color='gray', label='valid target')

plt.plot(np.arange(y_train.shape[0]+y_valid.shape[0],
                   y_train.shape[0]+y_test.shape[0]+y_test.shape[0]),
         y_test[:,ft], color='black', label='test target')

plt.plot(np.arange(y_train_pred.shape[0]),y_train_pred[:,ft], color='red',
         label='train prediction')

plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_valid_pred.shape[0]),
         y_valid_pred[:,ft], color='orange', label='valid prediction')

plt.plot(np.arange(y_train_pred.shape[0]+y_valid_pred.shape[0],
                   y_train_pred.shape[0]+y_valid_pred.shape[0]+y_test_pred.shape[0]),
         y_test_pred[:,ft], color='green', label='test prediction')

plt.title('past and future stock prices')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best')

plt.subplot(1,2,2)

plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_test.shape[0]),
         y_test[:,ft], color='black', label='test target')

plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_test_pred.shape[0]),
         y_test_pred[:,ft], color='green', label='test prediction')

plt.title('future stock prices')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best')

corr_price_development_train = np.sum(np.equal(np.sign(y_train[:,1]-y_train[:,0]),
            np.sign(y_train_pred[:,1]-y_train_pred[:,0])).astype(int)) / y_train.shape[0]
corr_price_development_valid = np.sum(np.equal(np.sign(y_valid[:,1]-y_valid[:,0]),
            np.sign(y_valid_pred[:,1]-y_valid_pred[:,0])).astype(int)) / y_valid.shape[0]
corr_price_development_test = np.sum(np.equal(np.sign(y_test[:,1]-y_test[:,0]),
            np.sign(y_test_pred[:,1]-y_test_pred[:,0])).astype(int)) / y_test.shape[0]

print('correct sign prediction for close - open price for train/valid/test: %.2f/%.2f/%.2f'%(
    corr_price_development_train, corr_price_development_valid, corr_price_development_test))

# -------------------------------------- End of Stock LSTM ---------------------------------------------------- #


# ------------------------------------------ Curve LSTM ---------------------------------------------------- #

# ---------------------------------------- End of Curve LSTM ------------------------------------------------ #

# ------------------------------------------ Reinforcement Learning ------------------------------------------------ #

# ---------------------------------------- End of Reinforcement Learning ------------------------------------------ #

# ------------------------------------------ New ------------------------------------------------ #
import numpy as np
import pandas as pd
import tensorflow as tf

n_window_size = 20            # 20 business days

# price to return normalization
spx = pd.read_csv('hist/SPX Index.csv', index_col=0, header=0)
spx = spx[['High', 'Low', 'Close']]
spx_close = spx[['Close']*3]
spx_close.columns = spx.columns
spx = (spx / spx_close.shift(1) - 1)*100.0      # in percentage
spx.dropna(inplace=True)            # shape = (4737, 3), from 1/4/2000 to 10/31/2018

# split between train and test, 90%/10%
total_size = spx.shape[0] - n_steps + 1             # 4718
train_set_size = int(np.round(total_size*0.9))      # 4246
test_set_size = total_size - train_set_size       # 472

x_train = np.zeros((train_set_size, n_steps-1, 3))      # shape = (4246, 19, 3)
y_train = np.zeros((train_set_size, 1))                 # shape = (4246, 1)
x_test = np.zeros((test_set_size, n_steps-1, 3))        # shape = (472, 19, 3)
y_test = np.zeros((test_set_size, 1))                   # shape = (472, 1)

for i in range(train_set_size):
    x_train[i, :, :] = spx.iloc[i:i+n_steps-1].values
    y_train[i, 0] = spx.iloc[i + n_steps - 1, 2]

for i in range(train_set_size, total_size):
    x_test[i-train_set_size, :, :] = spx.iloc[i:i+n_steps-1].values
    y_test[i-train_set_size, 0] = spx.iloc[i + n_steps - 1, 2]

# generate next batch
index_in_epoch = 0
perm_array = np.arange(x_train.shape[0])            # (4246,)
np.random.shuffle(perm_array)


# function to get the next batch; randomly draw 50 20d-windows
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

n_steps = n_window_size - 1            # 20 business days
n_inputs = 3            # HLC       # true range
n_outputs = 1           # C(t+1)
n_neurons = 200
n_layers = 2
learning_rate = 0.001
batch_size = 50
n_epochs = 100

tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

# TODO: add MLP

# use Basic RNN Cell
layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu) for layer in range(n_layers)]

# use Basic LSTM Cell
#layers = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.elu)
#          for layer in range(n_layers)]

# use LSTM Cell with peephole connections
#layers = [tf.contrib.rnn.LSTMCell(num_units=n_neurons,
#                                  activation=tf.nn.leaky_relu, use_peepholes = True)
#          for layer in range(n_layers)]

# use GRU cell
#layers = [tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.leaky_relu)
#          for layer in range(n_layers)]

# TODO: dropout wrapper; use name_scope
# http://androidkt.com/stock-price-prediction/
# https://lilianweng.github.io/lil-log/2017/07/08/predict-stock-prices-using-RNN-part-1.html

multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
# rnn_outputs contains the output tensors for each time step (?, 19, 200)
# states contains the final states of the network, (?, 200)x(2 layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
# add a densely-connected layer
y_pred = tf.layers.dense(states[-1], n_outputs)

loss = tf.reduce_mean(tf.square(y_pred - y))   # loss function = mean squared error
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

# run graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iteration in range(int(n_epochs*train_set_size/batch_size)):
        x_batch, y_batch = get_next_batch(batch_size) # fetch the next training batch
        #sess.run(training_op, feed_dict={X: x_batch, y: y_batch})
        [a1, a2, a3, a4] = sess.run([rnn_outputs, states, y_pred, training_op], feed_dict={X: x_batch, y: y_batch})
        if iteration % int(5*train_set_size/batch_size) == 0:
            mse_train = loss.eval(feed_dict={X: x_train, y: y_train})
            print('%.2f epochs: MSE train = %.6f'%(
                iteration*batch_size/train_set_size, mse_train))

    y_train_pred = sess.run(y_pred, feed_dict={X: x_train})
    y_test_pred = sess.run(y_pred, feed_dict={X: x_test})

# prediction


# multi-step prediction, next 5 days


# ---------------------------------------- End of New ------------------------------------------ #
