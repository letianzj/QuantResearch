'''
    Pruitt, George, and John R. Hill. Building Winning Trading Systems with Tradestation,+ Website. Vol. 542. John Wiley & Sons, 2012.
    By George Pruitt in 1996
    https://www.quantconnect.com/tutorials/strategy-library/the-dynamic-breakout-ii-strategy
    1. Adaptive Donchian lookback parameter to current market: In volatile markets, using longer lookback to avoid frequent in-and-out. In trending markets; using shorter lookback to follow the trend.
       if volatility changed x% from yesterday, change lookback x%. lookback falls between [20, 60]; Volatility can be ATR, stdev, or VIX
    2. Another condition is adaptive bollinger bands. It needs to be confirmed before openning a position.
    3. Adaptive moving average is used for stop loss.
'''
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


class DynamicBreakoutII(qt.StrategyBase):
    def __init__(self, lookback_days=20):
        super(DynamicBreakoutII, self).__init__()
        self.current_time = None
        self.lookback_days = lookback_days

    def on_tick(self, tick_event):
        self.current_time = tick_event.timestamp
        # print('Processing {}'.format(self.current_time))
        symbol = self.symbols[0]

        df_hist = self._data_board.get_hist_price(symbol, tick_event.timestamp)

        # wait for enough bars
        if df_hist.shape[0] <= self.lookback_days:
            return

        current_price = df_hist.iloc[-1].Close
        prev_price = df_hist.iloc[-2].Close
        current_size = self._position_manager.get_position_size(symbol)
        npv = self._position_manager.current_total_capital

        today_vol = np.std(df_hist.Close[-self.lookback_days:])
        yesterday_vol = np.std(df_hist.Close[-self.lookback_days-1:-1])
        delta_vol = (today_vol / yesterday_vol) / today_vol
        self.lookback_days = round(self.lookback_days * (1 + delta_vol), 0)
        self.lookback_days = int(min(max(self.lookback_days, 20), 60))

        hh = max(df_hist.High[-self.lookback_days-1:-1])
        ll = min(df_hist.Low[-self.lookback_days-1:-1])
        ma = np.average(df_hist.Close[-self.lookback_days:])
        sd = np.std(df_hist.Close[-self.lookback_days:])
        ub = ma + 2.0*sd
        lb = ma - 2.0*sd

        if current_size == 0:
            target_size = int(npv/ current_price)
            if current_price > ub:  # and current_price > hh:
                self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
                print(f'LONG ORDER SENT, price: {current_price:.2f}, ub: {ub:.2f}, hh: {hh:.2f}, size: {target_size}')
            elif current_price < lb:  # and current_price < ll:
                self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
                print(f'SHORT ORDER SENT, price: {current_price:.2f}, lb: {lb:.2f}, ll: {ll:.2f}, size: {-target_size}')
        # exit long if price < MA; exit short if price > MA
        elif current_size > 0:
            if current_price < ma:
                target_size = 0
                self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
                print(f'FLAT LONG ORDER SENT, price: {current_price:.2f}, ma: {ma:.2f}, size: {-current_size}')
        else:
            if current_price > ma:
                target_size = 0
                self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
                print(f'FLAT SHORT ORDER SENT, price: {current_price:.2f}, ma: {ma:.2f}, size: {-current_size}')


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
        for n_ in [20, 30, 40]:
                params_list.append({'lookback_days': n_})
        target_name = 'Sharpe ratio'
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for params in params_list:
            strategy = DynamicBreakoutII()
            strategy.set_capital(init_capital)
            strategy.set_symbols([symbol])
            backtest_engine = qt.BacktestEngine(test_start_date, test_end_date)
            backtest_engine.set_capital(init_capital)  # capital or portfolio >= capital for one strategy
            backtest_engine.add_data(symbol, data)
            strategy.set_params({'lookback_days': params['lookback_days']})
            backtest_engine.set_strategy(strategy)
            tag = (params['lookback_days'])
            p = multiprocessing.Process(target=parameter_search, args=(backtest_engine, tag, target_name, return_dict))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
        for k,v in return_dict.items():
            print(k, v)
    else:
        strategy = DynamicBreakoutII()
        strategy.set_capital(init_capital)
        strategy.set_symbols([symbol])
        # strategy.set_params(None)

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