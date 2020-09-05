'''
classical Bollinger Bands. Buy when price back up from lower bands; sell when middle is hit
Sell when price back down from upper bands; buy back when middle is hit
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
pd.set_option('display.max_columns', None)
display(HTML("<style>.container { width:100% !important; }</style>"))


class BollingerBands(qt.StrategyBase):
    def __init__(self,
            n=20, ndev=2.0
    ):
        super(BollingerBands, self).__init__()
        self.n = n
        self.ndev = ndev
        self.current_time = None

    def on_tick(self, tick_event):
        super().on_tick(tick_event)
        self.current_time = tick_event.timestamp
        # print('Processing {}'.format(self.current_time))
        symbol = self.symbols[0]

        df_hist = self._data_board.get_hist_price(symbol, tick_event.timestamp)

        # wait for enough bars
        if df_hist.shape[0] < self.n:
            return

        current_price = df_hist.iloc[-1].Close
        prev_price = df_hist.iloc[-2].Close
        current_size = self._position_manager.get_position_size(symbol)
        npv = self._position_manager.current_total_capital
        ub, mb, lb = talib.BBANDS(df_hist['Close'], timeperiod=self.n, nbdevup=self.ndev, nbdevdn=self.ndev, matype=0)
        # open long position; price backs up from below lower band
        if current_size <= 0 and current_price > lb[-1] and prev_price < lb[-2]:
            target_size = (int)(npv / current_price)
            self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
            print(f'{self.current_time}, npv {npv}, BUY ORDER SENT, {symbol}, Pre-Price: {prev_price:.2f}, '
                  f'Price: {current_price:.2f}, Pre-LB: {lb[-2]:.2f}, LB: {lb[-1]}, Size: {target_size}')
        # open short position; price backs down from above upper band
        elif current_size >= 0 and current_price < ub[-1] and prev_price > ub[-2]:
            target_size = -(int)(npv / current_price)
            self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
            print(f'{self.current_time}, npv {npv}, SELL ORDER SENT, {symbol}, Pre-Price: {prev_price:.2f}, '
                  f'Price: {current_price:.2f}, Pre-UB: {ub[-2]:.2f}, UB: {ub[-1]}, Size: {target_size}')
        # close short position
        elif current_price < mb[-1] and current_size < 0:
            target_size = 0
            self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
            print(f'{self.current_time}, npv {npv}, CLOSE SHORT ORDER SENT, {symbol}, Price: {current_price:.2f}, MB: {mb[-1]}, Size: {target_size}')
        # close long position
        elif current_price > mb[-1] and current_size > 0:
            target_size = 0
            self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
            print(f'{self.current_time}, npv {npv}, CLOSE LONG ORDER SENT, {symbol}, Price: {current_price:.2f}, MB: {mb[-1]}, Size: {target_size}')
        else:
            print(f'{self.current_time}, npv {npv}')


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
        for n_ in [10, 20, 30, 50, 100]:
            for nd_ in [1.0, 1.5, 2.0, 2.5]:
                params_list.append({'n': n_, 'ndev': nd_})
        target_name = 'Sharpe ratio'
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for params in params_list:
            strategy = BollingerBands()
            strategy.set_capital(init_capital)
            strategy.set_symbols([symbol])
            backtest_engine = qt.BacktestEngine(test_start_date, test_end_date)
            backtest_engine.set_capital(init_capital)  # capital or portfolio >= capital for one strategy
            backtest_engine.add_data(symbol, data)
            strategy.set_params({'n': params['n'], 'ndev': params['ndev']})
            backtest_engine.set_strategy(strategy)
            tag = (params['n'], params['ndev'])
            p = multiprocessing.Process(target=parameter_search, args=(backtest_engine, tag, target_name, return_dict))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
        for k,v in return_dict.items():
            print(k, v)
    else:
        strategy = BollingerBands()
        strategy.set_capital(init_capital)
        strategy.set_symbols([symbol])
        strategy.set_params({'n':20, 'ndev':2.0})

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
            f8 = plt.figure(8)
            pf.create_position_tear_sheet(strat_ret, df_positions)
            plt.show()
            f9 = plt.figure(9)
            pf.create_txn_tear_sheet(strat_ret, df_positions, df_trades)
            plt.show()
            f10 = plt.figure(10)
            pf.create_round_trip_tear_sheet(strat_ret, df_positions, df_trades)
            plt.show()