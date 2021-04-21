#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn import linear_model

def locate_consecutive_with_conditions(df, op, rhs):
    p = op(df, rhs)
    c = p.cumsum()
    d = c - c.mask(p).ffill().fillna(0).astype(int)
    return d

def calculate_half_life_of_time_series(hist_df):
    df_lag = hist_df.shift(1)
    df_delta = hist_df - df_lag
    lin_reg_model = linear_model.LinearRegression()
    df_delta = df_delta.values.reshape(len(df_delta), 1)  # sklearn needs (row, 1) instead of (row,)
    df_lag = df_lag.values.reshape(len(df_lag), 1)
    lin_reg_model.fit(df_lag[1:], df_delta[1:])  # skip first line nan
    half_life = -np.log(2) / lin_reg_model.coef_.item()
    return half_life