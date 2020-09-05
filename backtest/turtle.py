'''
    Richard Dennis
    https://www.quantopian.com/posts/turtle-trading-strategy#:~:text=Turtle%20trading%20is%20a%20well,of%20rules%20is%20more%20intricate.&text=This%20is%20a%20pretty%20fundamental%20strategy%20and%20it%20seems%20to%20work%20well.
    https://bigpicture.typepad.com/comments/files/turtlerules.pdf
    https://github.com/myquant/strategy/blob/master/Turtle/info.md
    https://zhuanlan.zhihu.com/p/161882477
    trend following
    entry: price > 20 day High
    add: for every 0.5 ATR, up to 3 times
    stop: < 2 ATR
    stop: < 10 day Low
    It makes investments in units: one price unit is one ATR; one size unit is 1% of asset / ATR.
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


class Turtle(qt.StrategyBase):
    def __init__(self, short_window=10, long_window=20):
        super(Turtle, self).__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.buy_count = 0
        self.buyprice = 0.0
        self.current_time = None

    def on_fill(self, fill_event):
        super().on_fill(fill_event)
        if fill_event.fill_size > 0:         # buy
            self.buyprice = fill_event.fill_price

    def on_tick(self, tick_event):
        self.current_time = tick_event.timestamp
        # print('Processing {}'.format(self.current_time))
        symbol = self.symbols[0]

        df_hist = self._data_board.get_hist_price(symbol, tick_event.timestamp)

        # wait for enough bars
        if df_hist.shape[0] <= self.long_window:
            return

        current_price = df_hist.iloc[-1].Close
        prev_price = df_hist.iloc[-2].Close
        current_size = self._position_manager.get_position_size(symbol)
        npv = self._position_manager.current_total_capital
        don_high = max(df_hist.High.iloc[-self.long_window-1:-1])            # 20d high
        don_low = max(df_hist.High.iloc[-self.short_window - 1:-1])          # 10d low
        TR = pd.concat([df_hist.High.iloc[-15:-1]-df_hist.Low.iloc[-15:-1],
                        abs((df_hist.Close.iloc[-15:].shift(-1) - df_hist.High.iloc[-15:-1]).dropna()),
                        abs((df_hist.Close.iloc[-15:].shift(-1) - df_hist.Low.iloc[-15:-1]).dropna())], axis=1).max(axis=1)
        ATR = np.average(TR)

        # Long
        if current_price > don_high and self.buy_count == 0:
            # one unit is 1% of total risk asset
            target_size = int(npv * 0.01 / ATR)
            self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
            print(f'LONG ORDER SENT, price: {current_price:.2f}, don_high: {don_high:.2f}')
            self.buy_count = 1
        # add; This is for futures; may go beyond notional; leverage is set to 4
        elif current_price > self.buyprice + 0.5 * ATR and self.buy_count > 0 and self.buy_count <= 3:
            target_size = int(npv * 0.01 / ATR)
            target_size += current_size        # on top of current size
            self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
            print(f'ADD LONG ORDER SENT, add time: {self.buy_count}, price: {current_price:.2f}, don_high: {don_high:.2f}')
            self.buy_count += 1
        # flat
        elif current_price < don_low and self.buy_count > 0:
            target_size = 0
            self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
            print(f'FLAT ORDER SENT, price: {current_price:.2f}, don_low: {don_low:.2f}')
            self.buy_count = 0
        # flat, stop loss
        elif current_price < (self.buyprice - 2 * ATR) and self.buy_count > 0:
            target_size = 0
            self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
            print(f'FLAT ORDER SENT, price: {current_price:.2f}, {self.buyprice:.2f}, 2ATR: {2 * ATR:.2f}')
            self.buy_count = 0


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
        for sw_ in [10, 20, 30, 40, 50]:
            for lw_ in [10, 20, 30, 40, 50]:
                if sw_ >= lw_:
                    continue
                params_list.append({'short_window': sw_, 'long_window': lw_})
        target_name = 'Sharpe ratio'
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for params in params_list:
            strategy = Turtle()
            strategy.set_capital(init_capital)
            strategy.set_symbols([symbol])
            backtest_engine = qt.BacktestEngine(test_start_date, test_end_date)
            backtest_engine.set_capital(init_capital)  # capital or portfolio >= capital for one strategy
            backtest_engine.add_data(symbol, data)
            strategy.set_params({'short_window': params['short_window'], 'long_window': params['long_window']})
            backtest_engine.set_strategy(strategy)
            tag = (params['short_window'], params['long_window'])
            p = multiprocessing.Process(target=parameter_search, args=(backtest_engine, tag, target_name, return_dict))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
        for k,v in return_dict.items():
            print(k, v)
    else:
        strategy = Turtle()
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