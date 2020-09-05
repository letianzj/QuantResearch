"""
The Dual Thrust trading algorithm is a famous strategy developed by Michael Chalek.
It is a breakout system, commonly used in futures, forex and equity markets.
The limits are based on today’s opening price plus or minus a certain percentage of recent trading range.
When the price breaks through the upper level, it will long, and when it breaks the lower level, it will short.
1. recent trading range is relatively stable, using four price points;
2. Percentage K1 and K2 can be asymmetric
https://www.quantconnect.com/tutorials/strategy-library/dual-thrust-trading-algorithm
Similar to quantconnect, got negative Sharpe -0.377.
It is an intraday breakout strategy, requires tickdata; holding position for a year is against the essence of this strategy.
Improvements: 1. profit target and stop loss. 2. confirmation e.g. MA5min>MA10min
"""
import os
import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timezone
import multiprocessing
import talib
import quanttrader as qt
import matplotlib.pyplot as plt
import empyrical as ep
import pyfolio as pf
# set browser full width
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


class DualThrust(qt.StrategyBase):
    def __init__(self,
            n=4, k1=0.5, k2=0.5
    ):
        super(DualThrust, self).__init__()
        self.n = n
        self.k1 = k1
        self.k2 = k2
        self.current_time = None

    def on_tick(self, tick_event):
        self.current_time = tick_event.timestamp
        # print('Processing {}'.format(self.current_time))
        symbol = self.symbols[0]

        df_hist = self._data_board.get_hist_price(symbol, tick_event.timestamp)

        # need n day trading range
        if df_hist.shape[0] < self.n:
            return

        high = df_hist.High.iloc[-self.n:]
        low = df_hist.Low.iloc[-self.n:]
        close = df_hist.Close.iloc[-self.n:]
        current_open = df_hist.Open.iloc[-1]
        current_price = df_hist.Close.iloc[-1]
        current_size = self._position_manager.get_position_size(symbol)
        npv = self._position_manager.current_total_capital

        HH, HC, LC, LL = max(high), max(close), min(close), min(low)
        signal_range = max(HH - LC, HC - LL)
        selltrig = current_open - self.k2 * signal_range
        buytrig = current_open + self.k1 * signal_range

        if current_price > buytrig:   # buy on upper break
            if current_size > 0:
                return
            target_size = int(npv / current_price)
            self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
            print(f'{self.current_time}, BUY ORDER SENT, {symbol}, Price: {current_price:.2f}, '
                  f'Buy trigger: {buytrig:.2f}, Size: {current_size},  Target Size: {target_size}')
        elif current_price < selltrig:   # sell on down break
            if current_size < 0:
                return
            target_size = -int(npv / current_price)
            self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
            print(f'{self.current_time}, SELL ORDER SENT, {symbol}, Price: {current_price:.2f}, '
                  f'Sell trigger: {selltrig:.2f}, Size: {current_size},  Target Size: {target_size}')


def parameter_search(engine, tag, target_name, return_dict):
    """
    This function should be the same for all strategies.
    The only reason not included in quanttrader is because of its dependency on pyfolio (to get perf_stats)
    """
    ds_equity, _, _ = engine.run()
    try:
        strat_ret = ds_equity.pct_change().dropna()
        perf_stats_strat = pf.timeseries.perf_stats(strat_ret)
        target_value = perf_stats_strat.loc[target_name]  # first table in tuple
    except KeyError:
        target_value = 0
    return_dict[tag] = target_value


if __name__ == '__main__':
    do_optimize = False
    run_in_jupyter = False
    symbol = 'SPX'
    benchmark = 'SPX'
    datapath = os.path.join('../data/', f'{symbol}.csv')
    data = qt.util.read_ohlcv_csv(datapath)
    init_capital = 100_000.0
    test_start_date = datetime(2010,1,1, 8, 30, 0, 0, pytz.timezone('America/New_York'))
    test_end_date = datetime(2019,12,31, 6, 0, 0, 0, pytz.timezone('America/New_York'))

    if do_optimize:          # parallel parameter search
        params_list = []
        for n_ in [3, 4, 5, 10]:
            for k_ in [0.4, 0.5, 0.6]:
                params_list.append({'n': n_, 'k1': k_, 'k2': k_})
        target_name = 'Sharpe ratio'
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for params in params_list:
            strategy = DualThrust()
            strategy.set_capital(init_capital)
            strategy.set_symbols([symbol])
            backtest_engine = qt.BacktestEngine(test_start_date, test_end_date)
            backtest_engine.set_capital(init_capital)  # capital or portfolio >= capital for one strategy
            backtest_engine.add_data(symbol, data)
            strategy.set_params({'n': params['n'], 'k1': params['k1'], 'k2': params['k2']})
            backtest_engine.set_strategy(strategy)
            tag = (params['n'], params['k1'], params['k2'])
            p = multiprocessing.Process(target=parameter_search, args=(backtest_engine, tag, target_name, return_dict))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
        for k,v in return_dict.items():
            print(k, v)
    else:
        strategy = DualThrust()
        strategy.set_capital(init_capital)
        strategy.set_symbols([symbol])
        strategy.set_params({'n':4, 'k1': 0.5, 'k2': 0.5})

        # Create a Data Feed
        backtest_engine = qt.BacktestEngine(test_start_date, test_end_date)
        backtest_engine.set_capital(init_capital)  # capital or portfolio >= capital for one strategy
        backtest_engine.add_data(symbol, data)
        backtest_engine.set_strategy(strategy)
        ds_equity, df_positions, df_trades = backtest_engine.run()
        # save to excel
        qt.util.save_one_run_results('./output', ds_equity, df_positions, df_trades)

        # ------------------------- Evaluation and Plotting -------------------------------------- #
        strat_ret = ds_equity.pct_change().dropna()
        strat_ret.name = 'strat'
        bm = qt.util.read_ohlcv_csv(os.path.join('../data/', f'{benchmark}.csv'))
        bm_ret = bm['Close'].pct_change().dropna()
        bm_ret.index = pd.to_datetime(bm_ret.index)
        bm_ret = bm_ret[strat_ret.index]
        bm_ret.name = 'benchmark'

        perf_stats_strat = pf.timeseries.perf_stats(strat_ret)
        perf_stats_all = perf_stats_strat
        perf_stats_bm = pf.timeseries.perf_stats(bm_ret)
        perf_stats_all = pd.concat([perf_stats_strat, perf_stats_bm], axis=1)
        perf_stats_all.columns = ['Strategy', 'Benchmark']

        drawdown_table = pf.timeseries.gen_drawdown_table(strat_ret, 5)
        monthly_ret_table = ep.aggregate_returns(strat_ret, 'monthly')
        monthly_ret_table = monthly_ret_table.unstack().round(3)
        ann_ret_df = pd.DataFrame(ep.aggregate_returns(strat_ret, 'yearly'))
        ann_ret_df = ann_ret_df.unstack().round(3)

        print('-------------- PERFORMANCE ----------------')
        print(perf_stats_all)
        print('-------------- DRAWDOWN ----------------')
        print(drawdown_table)
        print('-------------- MONTHLY RETURN ----------------')
        print(monthly_ret_table)
        print('-------------- ANNUAL RETURN ----------------')
        print(ann_ret_df)

        if run_in_jupyter:
            pf.create_full_tear_sheet(
                strat_ret,
                benchmark_rets=bm_ret,
                positions=df_positions,
                transactions=df_trades,
                round_trips=False)
            plt.show()
        else:
            f1 = plt.figure(1)
            pf.plot_rolling_returns(strat_ret, factor_returns=bm_ret)
            f1.show()
            f2 = plt.figure(2)
            pf.plot_rolling_volatility(strat_ret, factor_returns=bm_ret)
            f2.show()
            f3 = plt.figure(3)
            pf.plot_rolling_sharpe(strat_ret)
            f3.show()
            f4 = plt.figure(4)
            pf.plot_drawdown_periods(strat_ret)
            f4.show()
            f5 = plt.figure(5)
            pf.plot_monthly_returns_heatmap(strat_ret)
            f5.show()
            f6 = plt.figure(6)
            pf.plot_annual_returns(strat_ret)
            f6.show()
            f7 = plt.figure(7)
            pf.plot_monthly_returns_dist(strat_ret)
            plt.show()