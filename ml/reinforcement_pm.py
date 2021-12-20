#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gym trading env
Unlike live engine or backtest engine, where event loops are driven by live ticks or historical ticks,
here it is driven by step function, similar to
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
The sequence is
1. obs <- env.reset()      # env; obs = OHLCV + technical indicators + holdings
2. action <- pi(obs)       # agent; action = target weights
3. news_obs, reward, <- step(action)      # env
    3.a action to orders   # sum(holdings) * new weights
    3.b fill orders        # portfolio rotates into new holdings
repeat 2, and 3 to interact between agent and env
"""
import numpy as np
import pandas as pd
from quanttrader import PortfolioEnv


def load_data():
    from datetime import timedelta
    import ta

    sd = '2015'
    ed = '2020'
    syms = ['SPY', 'AAPL']
    max_price_scaler = 5_000.0
    max_volume_scaler = 1.5e10
    df_obs = pd.DataFrame()             # observation
    df_exch = pd.DataFrame()            # exchange; for order match

    for sym in syms:
        df = pd.read_csv(f'../data/{sym}.csv', index_col=0)
        df.index = pd.to_datetime(df.index) + timedelta(hours=15, minutes=59, seconds=59)
        df = df[sd:ed]

        df_exch = pd.concat([df_exch, df['Close'].rename(sym)], axis=1)

        df['Open'] = df['Adj Close'] / df['Close'] * df['Open'] / max_price_scaler
        df['High'] = df['Adj Close'] / df['Close'] * df['High'] / max_price_scaler
        df['Low'] = df['Adj Close'] / df['Close'] * df['Low'] / max_price_scaler
        df['Volume'] = df['Adj Close'] / df['Close'] * df['Volume'] / max_volume_scaler
        df['Close'] = df['Adj Close'] / max_price_scaler
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = [f'{sym}:{c.lower()}' for c in df.columns]

        macd = ta.trend.MACD(close=df[f'{sym}:close'])
        df[f'{sym}:macd'] = macd.macd()
        df[f'{sym}:macd_diff'] = macd.macd_diff()
        df[f'{sym}:macd_signal'] = macd.macd_signal()

        rsi = ta.momentum.RSIIndicator(close=df[f'{sym}:close'])
        df[f'{sym}:rsi'] = rsi.rsi()

        bb = ta.volatility.BollingerBands(close=df[f'{sym}:close'], window=20, window_dev=2)
        df[f'{sym}:bb_bbm'] = bb.bollinger_mavg()
        df[f'{sym}:bb_bbh'] = bb.bollinger_hband()
        df[f'{sym}:bb_bbl'] = bb.bollinger_lband()

        atr = ta.volatility.AverageTrueRange(high=df[f'{sym}:high'], low=df[f'{sym}:low'], close=df[f'{sym}:close'])
        df[f'{sym}:atr'] = atr.average_true_range()

        df_obs = pd.concat([df_obs, df], axis=1)

    return df_obs, df_exch


if __name__ == '__main__':
    look_back = 10
    cash = 100_000.0
    max_nav_scaler = cash

    df_obs, df_exch = load_data()

    trading_env = PortfolioEnv(df_obs, df_exch)
    trading_env.set_cash(cash)
    trading_env.set_commission(0.0001)
    trading_env.set_steps(n_lookback=10, n_warmup=50, n_maxsteps=250)
    trading_env.set_feature_scaling(max_nav_scaler)
    o1 = trading_env.reset()

    # trading_env._current_step = look_back-1        # ignore randomness
    while True:
        action = trading_env.action_space.sample()
        o2, reward, done, info = trading_env.step(action)
        print(action, reward * max_nav_scaler, info)
        if done:
            break