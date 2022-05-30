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
plt.scatter(x, y, color='blue')

# initial belief
sigma_e = 3.0            # make it known to avoid inverse gamma complexity
a_0 = 0.5
b_0 = 0.5
sigma_a_0 = 0.5
sigma_b_0 = 0.5
beta_0 = np.array([[a_0], [b_0]])
sigma_beta_0 = np.array([[sigma_a_0*sigma_a_0, 0], [0, sigma_b_0*sigma_b_0]])

beta_recorder = []           # record parameter beta
beta_recorder.append(beta_0)
for pair in range(250):       # 500 points means 250 pairs
    x1 = x[pair*2]
    x2 = x[pair*2+1]
    y1 = y[pair*2]
    y2 = y[pair*2+1]
    mu_y = np.array([[(x1*y2-x2*y1)/(x1-x2)], [(y1-y2)/(x1-x2)]])
    sigma_y = np.array([[(np.square(x1/(x1-x2))+np.square(x2/(x1-x2)))*np.square(sigma_e),0],
                             [0,2*np.square(sigma_e/(x1-x2))]])
    sigma_beta_1 = np.linalg.inv(np.linalg.inv(sigma_beta_0)+np.linalg.inv(sigma_y))
    beta_1 = sigma_beta_1.dot(np.linalg.inv(sigma_beta_0).dot(beta_0) + np.linalg.inv(sigma_y).dot(mu_y))

    # assign beta_1 to beta_0
    beta_0 = beta_1
    sigma_beta_0 = sigma_beta_1
    beta_recorder.append(beta_0)

print('parameters: %.7f, %.7f' %(beta_0[0], beta_0[1]))

# plot the Beyesian dynamics
xfit = np.linspace(0, 10, sample_size)
ytrue = 2.0 * xfit + 1.0       # we know the true value of slope and intercept
plt.plot(xfit, ytrue, label='true line', linewidth=3)
y0 = beta_recorder[0][1] * xfit + beta_recorder[0][0]
plt.plot(xfit, y0, label='initial belief', linewidth=1)
y1 = beta_recorder[1][1] * xfit + beta_recorder[1][0]
plt.plot(xfit, y1, label='1st update', linewidth=1)
y10 = beta_recorder[10][1] * xfit + beta_recorder[10][0]
plt.plot(xfit, y10, label='10th update', linewidth=1)
y100 = beta_recorder[100][1] * xfit + beta_recorder[100][0]
plt.plot(xfit, y100, label='100th update', linewidth=1)
plt.legend()
plt.show()