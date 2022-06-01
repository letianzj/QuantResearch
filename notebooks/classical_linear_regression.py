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

# normal equation to estimate the model parameters
X = np.vstack((np.ones(sample_size), x)).T
params_closed_form = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
print('parameters: %.7f, %.7f' %(params_closed_form[0], params_closed_form[1]))

from sklearn.linear_model import LinearRegression
# The next two lines does the regression
lm_model = LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
lm_model.fit(x.reshape(-1,1), y)        # fit() expects 2D array
print('parameters: %.7f, %.7f' %(lm_model.intercept_, lm_model.coef_))

# present the graph
xfit = np.linspace(0, 10, sample_size)
yfit = lm_model.predict(xfit.reshape(-1,1))
ytrue = 2.0 * xfit + 1.0       # we know the true value of slope and intercept
plt.scatter(x, y, color='blue')
plt.plot(xfit, yfit, color='red', label='fitted line', linewidth=3)
plt.plot(xfit, ytrue, color='green', label='true line', linewidth=3)
plt.legend()

# R-Square
r_square = lm_model.score(x.reshape(-1,1), y)
print('R-Square %.7f' %(r_square))

from scipy.stats.stats import pearsonr
# The square root of R-Square is correlation coefficient
print('Its square root is Pearson correlation coefficient: %.7f == %.7f' %(np.sqrt(r_square), pearsonr(x, y)[0]))