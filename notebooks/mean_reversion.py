#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date

hist_file = os.path.join('hist/', '%s.csv' % 'USDCAD Curncy')
usd_cad = pd.read_csv(hist_file, header=0, parse_dates=True, sep=',', index_col=0)
usd_cad = usd_cad['Price']
usd_cad.name = 'USDCAD Curncy'
plt.plot(usd_cad.index, usd_cad, '-')
plt.xlabel('Date')
plt.ylabel('USDCAD')
plt.show()

###################################################### ADF test #####################################################
import statsmodels.tsa.stattools as ts
# H0: beta==1 or random walk
adf_statistic = ts.adfuller(usd_cad, 1)           # lag = 1
print('Augmented Dickey Fuller test statistic =',adf_statistic[0])   #  -2.0188553406859833
print('Augmented Dickey Fuller p-value =',adf_statistic[1])   # 0.27836737105308673
print('Augmented Dickey Fuller # of samples =',adf_statistic[3])  # 1599
# {'1%': -3.4344462031760283, '5%': -2.8633492329988335, '10%': -2.5677331999518147}
print('Augmented Dickey Fuller 1%, 5% and 10% critical values =',adf_statistic[4])


################################################### Hurst Exponent ##################################################
def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2.0

print("Hurst(USDCAD):   %s" % hurst(np.log(usd_cad)))

################################################# Variance Ratio #####################################################
def normcdf(X):
    (a1, a2, a3, a4, a5) = (0.31938153, -0.356563782, 1.781477937, -1.821255978, 1.330274429)
    L = abs(X)
    K = 1.0 / (1.0 + 0.2316419 * L)
    w = 1.0 - 1.0 / np.sqrt(2 * np.pi) * np.exp(-L * L / 2.) * (
                a1 * K + a2 * K * K + a3 * pow(K, 3) + a4 * pow(K, 4) + a5 * pow(K, 5))
    if X < 0:
        w = 1.0 - w
    return w

def vratio(a, lag=2, cor='hom'):
    t = (np.std((a[lag:]) - (a[1:-lag + 1]))) ** 2
    b = (np.std((a[2:]) - (a[1:-1]))) ** 2

    n = float(len(a))
    mu = sum(a[1:len(a)] - a[:-1]) / n
    m = (n - lag + 1) * (1 - lag / n)
    #   print mu, m, lag
    b = sum(np.square(a[1:len(a)] - a[:len(a) - 1] - mu)) / (n - 1)
    t = sum(np.square(a[lag:len(a)] - a[:len(a) - lag] - lag * mu)) / m
    vratio = t / (lag * b)

    la = float(lag)

    if cor == 'hom':
        varvrt = 2 * (2 * la - 1) * (la - 1) / (3 * la * n)


    elif cor == 'het':
        varvrt = 0
        sum2 = sum(np.square(a[1:len(a)] - a[:len(a) - 1] - mu))
        for j in range(lag - 1):
            sum1a = np.square(a[j + 1:len(a)] - a[j:len(a) - 1] - mu)
            sum1b = np.square(a[1:len(a) - j] - a[0:len(a) - j - 1] - mu)
            sum1 = np.dot(sum1a, sum1b)
            delta = sum1 / (sum2 ** 2)
            varvrt = varvrt + ((2 * (la - j) / la) ** 2) * delta

    zscore = (vratio - 1) / np.sqrt(float(varvrt))
    pval = normcdf(zscore)

    return vratio, zscore, pval

#  (1.043812391881447, 0.23398177899239425, 0.5925003942830439)
vratio(np.log(usd_cad.values), cor='het', lag=20)

###################################################### Half-Life #####################################################
from sklearn import linear_model
df_close = usd_cad.to_frame()
df_lag = df_close.shift(1)
df_delta = df_close - df_lag
lin_reg_model = linear_model.LinearRegression()
df_delta = df_delta.values.reshape(len(df_delta),1)                    # sklearn needs (row, 1) instead of (row,)
df_lag = df_lag.values.reshape(len(df_lag),1)
lin_reg_model.fit(df_lag[1:], df_delta[1:])                           # skip first line nan
half_life = -np.log(2) / lin_reg_model.coef_.item()
print ('Half life:       %s' % half_life)           #  260.65118856658813

################################################## Linear Scaling-in #################################################
# in source/straetgy/mystrategy folder