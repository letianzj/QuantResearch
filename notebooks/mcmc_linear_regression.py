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

import scipy.stats
# don't forget to generate the 500 random samples as in the previous post
sigma_e = 3.0

# Similar to last post, let's initially believe that a, b follow Normal distribution with mean 0.5 and standard deviation 0.5
# it returns the probability of seeing beta under this belief
def prior_probability(beta):
    a = beta[0]     # intercept
    b = beta[1]     # slope
    a_prior = scipy.stats.norm(0.5, 0.5).pdf(a)
    b_prior = scipy.stats.norm(0.5, 0.5).pdf(b)
    # log probability transforms multiplication to summation
    return np.log(a) + np.log(b)

# Given beta, the likehood of seeing x and y
def likelihood_probability(beta):
    a = beta[0]     # intercept
    b = beta[1]     # slope
    y_predict = a  + b * x
    single_likelihoods = scipy.stats.norm(y_predict, sigma_e).pdf(y)        # we know sigma_e is 3.0
    return np.sum(np.log(single_likelihoods))

# We don't need to know the denominator of support f(y)
# as it will be canceled out in the acceptance ratio
def posterior_probability(beta):
    return likelihood_probability(beta) + prior_probability(beta)

# jump from beta to new beta
# proposal function is Gaussian centered at beta
def proposal_function(beta):
    a = beta[0]
    b = beta[1]
    a_new = np.random.normal(a, 0.5)
    b_new = np.random.normal(b, 0.5)
    beta_new = [a_new, b_new]
    return beta_new

# run the Monte Carlo
beta_0 = [0.5, 0.5]        # start value
results = np.zeros([50000,2])            # record the results
results[0,0] = beta_0[0]
results[0, 1] = beta_0[1]
for step in range(1, 50000):               # loop 50,000 times
    print('step: {}'.format(step))

    beta_old = results[step-1, :]
    beta_proposal = proposal_function(beta_old)

    # Use np.exp to restore from log numbers
    prob = np.exp(posterior_probability(beta_proposal) - posterior_probability(beta_old))

    if np.random.uniform(0,1) < prob:
        results[step, :] = beta_proposal    # jump
    else:
        results[step, :] = beta_old         # stay

burn_in = 10000
beta_posterior = results[burn_in:, :]
print(beta_posterior.mean(axis=0))        # use average as point estimates

# present the results
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.hist(beta_posterior[:,0], bins=20, color='blue')
ax1.axvline(beta_posterior.mean(axis=0)[0], color='red', linestyle='dashed', linewidth=2)
ax1.title.set_text('Posterior -- Intercept')
ax2 = fig.add_subplot(122)
ax2.hist(beta_posterior[:,1], bins=20, color='blue')
ax2.axvline(beta_posterior.mean(axis=0)[1], color='red', linestyle='dashed', linewidth=2)
ax2.title.set_text('Posterior -- Slope')
plt.show()

