#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from datetime import datetime, date
import quandl

assets = ['AAPL',       # Apple
          'KO',         # Coca-Cola
          'DIS',        # Disney
          'XOM',        # Exxon Mobil
          'JPM',        # JPMorgan Chase
          'MCD',        # McDonald's
          'WMT']         # Walmart

# download historical data from quandl
hist_data = {}
for asset in assets:
    data = quandl.get('wiki/'+asset, start_date='2015-01-01', end_date='2017-12-31', authtoken='ay68s2CUzKbVuy8GAqxj')
    hist_data[asset] = data['Adj. Close']
hist_data = pd.concat(hist_data, axis=1)

# calculate historical log returns
hist_return = np.log(hist_data / hist_data.shift())
hist_return = hist_return.dropna()

# find historical mean, covriance, and correlation
hist_mean = hist_return.mean(axis=0).to_frame()
hist_mean.columns = ['mu']
hist_cov = hist_return.cov()
hist_corr = hist_return.corr()
print(hist_mean.transpose())
print(hist_cov)
print(hist_corr)

# construct random portfolios
n_portfolios = 5000
#set up array to hold results
port_returns = np.zeros(n_portfolios)
port_stdevs = np.zeros(n_portfolios)

for i in range(n_portfolios):
    w = np.random.rand(len(assets))        # random weights
    w = w / sum(w)                         # weights sum to 1
    port_return = np.dot(w.T, hist_mean.as_matrix()) * 250         # annualize; 250 business days
    port_stdev = np.sqrt(np.dot(w.T, np.dot(hist_cov, w))) * np.sqrt(250)  # annualize; 250 business days
    port_returns[i] = port_return
    port_stdevs[i] = port_stdev

plt.plot(port_stdevs, port_returns, 'o', markersize=6)
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.title('Return and Standard Deviation of Randomly Generated Portfolios')
plt.show()

# Global Minimum Variance (GMV) -- closed form
hist_cov_inv = - np.linalg.inv(hist_cov)
one_vec = np.ones(len(assets))
w_gmv = np.dot(hist_cov_inv, one_vec) / (np.dot(np.transpose(one_vec), np.dot(hist_cov_inv, one_vec)))
w_gmv_df = pd.DataFrame(data = w_gmv).transpose()
w_gmv_df.columns = assets
stdev_gmv = np.sqrt(np.dot(w_gmv.T, np.dot(hist_cov, w_gmv))) * np.sqrt(250)
print(w_gmv_df)
print(stdev_gmv)

# Global Minimum Variance (GMV) -- numerical
P = matrix(hist_cov.as_matrix())
q = matrix(np.zeros((len(assets), 1)))
A = matrix(1.0, (1, len(assets)))
b = matrix(1.0)
w_gmv_v2 = np.array(solvers.qp(P, q, A=A, b=b)['x'])
w_gmv_df_v2 = pd.DataFrame(w_gmv_v2).transpose()
w_gmv_df_v2.columns = assets
stdev_gmv_v2 = np.sqrt(np.dot(w_gmv_v2.T, np.dot(hist_cov, w_gmv_v2))) * np.sqrt(250)
print(w_gmv_df_v2)
print(np.asscalar(stdev_gmv_v2))

# Maximum return -- closed form
mu_o = np.asscalar(np.max(hist_mean))   # MCD
A = np.matrix([[np.asscalar(np.dot(hist_mean.T,np.dot(hist_cov_inv,hist_mean))),
                np.asscalar(np.dot(hist_mean.T,np.dot(hist_cov_inv,one_vec)))],
               [np.asscalar(np.dot(hist_mean.T,np.dot(hist_cov_inv,one_vec))),
                np.asscalar(np.dot(one_vec.T,np.dot(hist_cov_inv,one_vec)))]])
B = np.hstack([np.array(hist_mean),one_vec.reshape(len(assets),1)])
y = np.matrix([mu_o, 1]).T
w_max_ret = np.dot(np.dot(np.dot(hist_cov_inv, B),  np.linalg.inv(A)),y)
w_max_ret_df = pd.DataFrame(w_max_ret).T
w_max_ret_df.columns = assets
print(w_max_ret_df)

# Maximum return -- numerical
P = matrix(hist_cov.as_matrix())
q = matrix(np.zeros((len(assets), 1)))
A = matrix(np.hstack([np.array(hist_mean),one_vec.reshape(len(assets),1)]).transpose())
b = matrix([mu_o,1])
w_max_ret_v2 = np.array(solvers.qp(P, q, A=A, b=b)['x'])
w_max_ret_df_v2 = pd.DataFrame(w_max_ret_v2).transpose()
w_max_ret_df_v2.columns = assets
print(w_max_ret_df_v2)

# efficient frontier
N = 100
ef_left = np.asscalar(min(hist_mean.as_matrix()))          # minimum return
ef_right = np.asscalar(max(hist_mean.as_matrix()))         # maximum return
target_returns = np.linspace(ef_left, ef_right, N)         # N target returns
optimal_weights = [ solvers.qp(P, q, A=A, b=matrix([t,1]))['x'] for t in target_returns ]    # QP solver
ef_returns = [ np.asscalar(np.dot(w.T, hist_mean.as_matrix())*250) for w in optimal_weights ]         #a nnualized
ef_risks = [ np.asscalar(np.sqrt(np.dot(w.T, np.dot(hist_cov, w)) * 250)) for w in optimal_weights ]

plt.plot(port_stdevs, port_returns, 'o', markersize=6, label='Candidate Market Portfolio')
plt.plot(ef_risks, ef_returns, 'y-o', color='green', markersize=8, label='Efficient Frontier')
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier and Candidate Portfolios')
plt.legend(loc='best')
plt.show()

transition_data = pd.DataFrame(optimal_weights)
transition_data.columns = assets
plt.stackplot(range(50), transition_data.iloc[:50,:].T, labels=assets)             # the other half has negative weights
plt.legend(loc='upper left')
plt.margins(0, 0)
plt.title('Allocation Transition Matrix')
plt.show()

# Maximum sharpe -- closed form
r_f = 0.01
w_sharpe = np.dot(hist_cov_inv, hist_mean.as_matrix()-r_f/250) / np.dot(one_vec, np.dot(hist_cov_inv, hist_mean.as_matrix()-r_f/250))
w_sharpe_df = pd.DataFrame(w_sharpe).T
w_sharpe_df.columns = assets
print(w_sharpe_df)
print(mu_sharpe := np.dot(w_sharpe.T, hist_mean.as_matrix()) * 250)
print(stdev_sharpe := np.sqrt(np.dot(w_sharpe.T, np.dot(hist_cov, w_sharpe))) * np.sqrt(250))
print(sharpe_ratio := (mu_sharpe-r_f)/stdev_sharpe)


from scipy.optimize import minimize

fun = lambda w: -1 * np.dot(w.T, hist_mean.as_matrix()*250-r_f) / np.sqrt(np.dot(w.T, np.dot(hist_cov*250, w)))
cons = ({'type': 'eq', 'fun': lambda w:  np.dot(w.T, one_vec)-1})
res = minimize(fun, w_gmv, method='SLSQP', constraints=cons)
w_sharpe_v2 = res['x']
w_sharpe_v2_df = pd.DataFrame(w_sharpe_v2).T
w_sharpe_v2_df.columns = assets
print(w_sharpe_v2_df)
print(mu_sharpe_v2 := np.dot(w_sharpe_v2.T, hist_mean.as_matrix()) * 250)
print(stdev_sharpe_v2 := np.sqrt(np.dot(w_sharpe_v2.T, np.dot(hist_cov, w_sharpe_v2))) * np.sqrt(250))
print(sharpe_ratio_v2 := (mu_sharpe-r_f)/stdev_sharpe)
