#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, date
import quandl

assets = ['AAPL',       # Apple
          'KO',         # Coca-Cola
          'DIS',        # Disney
          'XOM',        # Exxon Mobil
          'JPM',        # JPMorgan Chase
          'MCD',        # McDonald's
          'WMT']         # Walmart

holdings = [100,200,300,400,500,600,700]          # number of shares in each assets

# download historical data from quandl
hist_data = {}
for asset in assets:
    data = quandl.get('wiki/'+asset, start_date='2015-01-01', end_date='2017-12-31', authtoken='ay68s2CUzKbVuy8GAqxj')
    hist_data[asset] = data['Adj. Close']
hist_data = pd.concat(hist_data, axis=1)

# calculate historical log returns
hist_return = np.log(hist_data / hist_data.shift())
hist_return = hist_return.dropna()

port_cov = hist_return.cov()             # portfolio covariance matrix
port_corr = hist_return.corr()           # portfolio correlation matrix

V_i = hist_data.iloc[-1] * holdings        # dollar value as of end_date
V_i = V_i.as_matrix()                        # convert to vector
V_p = V_i.sum()                          # dollar value of the portfolio

z = norm.ppf(0.95, 0, 1)                # z value
sigma_p = np.sqrt(np.dot(V_i.T, np.dot(port_cov.as_matrix(),V_i)))    # note it's in dollar amount
VaR_p = z * sigma_p                      # portfolio VaR

sigma_i = np.sqrt(np.diag(port_cov.as_matrix()))        # individual asset
VaR_i = z * sigma_i * V_i

cov_ip = np.dot(port_cov.as_matrix(), V_i)/V_p               # covariance
beta_i = cov_ip / (sigma_p*sigma_p/V_p/V_p)                  # beta
MVar_i = VaR_p/V_p*beta_i                                    # marginal var

CVaR_i = MVar_i * V_i                                        # component var
CVaR_i_df = pd.DataFrame(data=np.column_stack((V_i, V_i/V_p, CVaR_i, CVaR_i/VaR_p, beta_i)))
CVaR_i_df.index = assets
CVaR_i_df.columns = ['Position ($)', 'Position (%)','CVaR ($)','CVaR (%)', 'Beta']
print(CVaR_i_df)