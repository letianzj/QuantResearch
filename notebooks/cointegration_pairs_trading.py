#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date

#################################################### Data #####################################################
hist_file = os.path.join('hist/', '%s.csv' % 'EWA US Equity')
ewa_price = pd.read_csv(hist_file, header=0, parse_dates=True, sep=',', index_col=0)
ewa_price = ewa_price['Price']
ewa_price.name = 'EWA US Equity'

hist_file = os.path.join('hist/', '%s.csv' % 'EWC US Equity')
ewc_price = pd.read_csv(hist_file, header=0, parse_dates=True, sep=',', index_col=0)
ewc_price = ewc_price['Price']
ewc_price.name = 'EWC US Equity'

data = pd.concat([ewa_price, ewc_price], axis=1)
# print(data[data.isnull().any(axis=1)])
data.dropna(axis=0, how='any',inplace=True)

from sklearn.linear_model import LinearRegression
# The next two lines does the regression
lm_model = LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
lm_model.fit(data['EWA US Equity'].values.reshape(-1,1), data['EWC US Equity'].values)        # fit() expects 2D array
print('parameters: %.7f, %.7f' %(lm_model.intercept_, lm_model.coef_))

# present the graph
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].set_title('EWA vs EWC')
ax[0].plot(data)
yfit = lm_model.coef_ * data['EWA US Equity'] + lm_model.intercept_
y_residual = data['EWC US Equity'] - yfit
ax[1].set_title('Regression Residual')
ax[1].plot(y_residual)
plt.show()

from scipy.stats.stats import pearsonr
print('Pearson correlation coefficient:%.7f' %(pearsonr(data['EWA US Equity'], data['EWC US Equity'])[0]))
####################################### CADF #####################################################
import statsmodels.tsa.stattools as ts
ts.adfuller(y_residual, 1)           # lag = 1
# (-3.667485117146333,
#  0.0045944586170011716,
#  1,
#  4560,
#  {'1%': -3.431784865122899,
#   '5%': -2.8621740417619224,
#   '10%': -2.5671075035106954},
#  625.5003218990623)

lm_model = LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
lm_model.fit(data['EWC US Equity'].values.reshape(-1,1), data['EWA US Equity'].values)        # fit() expects 2D array
print('parameters: %.7f, %.7f' %(lm_model.intercept_, lm_model.coef_))
yfit = lm_model.coef_ * data['EWC US Equity'] + lm_model.intercept_
y_residual = data['EWA US Equity'] - yfit
ts.adfuller(y_residual, 1)           # lag = 1
# statistic = -3.797221868633519

####################################### Johansen #####################################################
from statsmodels.tsa.vector_ar.vecm import coint_johansen

jh_results = coint_johansen(data, 0, 1)             # 0 - constant term; 1 - log 1
print(jh_results.lr1)                           # dim = (n,) Trace statistic
print(jh_results.cvt)                           # dim = (n,3) critical value table (90%, 95%, 99%)
print(jh_results.evec)                          # dim = (n, n), columnwise eigen-vectors
v1 = jh_results.evec[:, 0]
v2 = jh_results.evec[:, 1]

# [21.44412674  3.64194243]                 # trace statistic
# [[13.4294 15.4943 19.9349]                # r = 0 critical values
#  [ 2.7055  3.8415  6.6349]]               # r <= 1 critical values
# [[ 0.53474958  0.02398649]                # eigenvectors
#  [-0.45169106  0.12036402]]