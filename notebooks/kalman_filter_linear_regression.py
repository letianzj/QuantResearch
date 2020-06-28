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

sigma_e = 3.0

# initial value
theta_0_0 = np.array([[0.5], [0.5]])         # 2x1 array
W = np.array([[0.5, 0], [0, 0.5]])          # 2x2 array
P_0_0 = W

results = np.zeros([250, 2])
for k in range(250):          # 250 pairs
    print('step {}'.format(k))
    # A-Priori prediction
    # first step, let k = 1
    theta_1_0 = theta_0_0
    P_1_0 = P_0_0 + W

    # After observing two points (x1, y1) and (x2, y2)
    x1 = x[2*k+0]
    x2 = x[2*k+1]
    y1 = y[2*k+0]
    y2 = y[2*k+1]
    y_1 = np.array([y1, y2]).reshape(2,1)
    F_1 = np.array([[1, x1], [1, x2]])
    y_1_tilde = y_1 - np.dot(F_1, theta_1_0)

    # residual covariance
    V_1 = np.array([[sigma_e, 0], [0, sigma_e]])
    S_1 = np.dot(np.dot(F_1, P_1_0), np.transpose(F_1)) + V_1

    # Kalman Gain
    K_1 = np.dot(np.dot(P_1_0, np.transpose(F_1)), np.linalg.inv(S_1))

    # Posterior
    theta_1_1 = theta_1_0 + np.dot(K_1, y_1_tilde)
    P_1_1 = np.eye(2) - np.dot(np.dot(K_1, F_1), P_1_0)

    # assign for next iteration
    results[k, :] = theta_1_1.reshape(2,)
    theta_0_0 = theta_1_1
    P_0_0 = P_1_1

print(results.mean(axis=0))     # intercept: 0.6694;   slope: 1.9926

# present the results
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(np.linspace(1, 250, num=250), results[:, 0])
ax1.title.set_text('Hidden State Evolution -- Intercept')
ax2 = fig.add_subplot(122)
plt.plot(np.linspace(1, 250, num=250), results[:, 1])
ax2.title.set_text('Hidden State Evolution -- Slope')
plt.show()