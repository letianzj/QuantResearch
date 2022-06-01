#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
from datetime import datetime, date
from hmmlearn.hmm import GaussianHMM

#################################################### Viterbi #####################################################
# https://en.wikipedia.org/wiki/Viterbi_algorithm
obs = ('happy', 'happy', 'happy')
states = ('Up', 'Down')
start_p = {'Up': 0.5, 'Down': 0.5}
trans_p = {
    'Up' : {'Up': 0.8, 'Down': 0.2},
    'Down' : {'Up': 0.3, 'Down': 0.7}
}
emit_p = {
    'Up' : {'happy': 0.9, 'unhappy': 0.1},
    'Down' : {'happy': 0.4, 'unhappy': 0.6}
}

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = max(V[t-1][prev_st]["prob"]*trans_p[prev_st][st] for prev_st in states)
            for prev_st in states:
                if V[t-1][prev_st]["prob"] * trans_p[prev_st][st] == max_tr_prob:
                    max_prob = max_tr_prob * emit_p[st][obs[t]]
                    V[t][st] = {"prob": max_prob, "prev": prev_st}
                    break
    for line in dptable(V):
        print(line)
    opt = []
    # The highest probability
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]
    print('The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob)

def dptable(V):
        # Print a table of steps from dictionary
        yield " ".join(("%12d" % i) for i in range(len(V)))
        for state in V[0]:
            yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)

viterbi(obs, states, start_p, trans_p, emit_p)

#################################################### Data #####################################################
# https://hmmlearn.readthedocs.io/en/latest/auto_examples/plot_hmm_stock_analysis.html
hist_file = os.path.join('hist/', '%s.csv' % 'SPX Index')
spx_price = pd.read_csv(hist_file, header=0, parse_dates=True, sep=',', index_col=0)
spx_price = spx_price['Close']
spx_price.name = 'SPX Index'
spx_ret = spx_price.shift(1)/ spx_price[1:] - 1
spx_ret.dropna(inplace=True)
#spx_ret = spx_ret * 1000.0
rets = np.column_stack([spx_ret])

# Create the Gaussian Hidden markov Model and fit it
# to the SPY returns data, outputting a score
hmm_model = GaussianHMM(
    n_components=3,                     # number of states
    covariance_type="full",             # full covariance matrix vs diagonal
    n_iter=1000                         # number of iterations
).fit(rets)

print("Model Score:", hmm_model.score(rets))

# Plot the in sample hidden states closing values
# Predict the hidden states array
hidden_states = hmm_model.predict(rets)

print('Percentage of hidden state 1 = %f' % (sum(hidden_states)/len(hidden_states)))

print("Transition matrix")
print(hmm_model.transmat_)

print("Means and vars of each hidden state")
for i in range(hmm_model.n_components):                   # 0 is down, 1 is up
    print("{0}th hidden state".format(i))
    print("mean = ", hmm_model.means_[i])
    print("var = ", np.diag(hmm_model.covars_[i]))

fig, axs = plt.subplots(hmm_model.n_components, sharex=True, sharey=True)
colours = cm.rainbow(np.linspace(0, 1, hmm_model.n_components))
for i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    mask = hidden_states == i
    ax.plot_date(spx_ret.index[mask], spx_price.loc[spx_ret.index][mask], ".", linestyle='none', c=colour)
    ax.set_title("{0}th hidden state".format(i))

    # Format the ticks.
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.grid(True)

plt.show()