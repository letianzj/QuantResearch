"""
It is observed that if last trade is profitable, next trade would more likely be a loss.
Then why not create a ghost trader on the same strategy; and trade only when the ghost trader's a loss.
Elements: two moving averages; rsi; donchain channel
conditions: 1. long if short MA > long MA, rsi lower than overbought 70, new high
            2. short if short MA < long MA, ris higher than oversold 30, new low
exit:       1. exit long if lower than donchian lower band
            2. exit short if higher than donchian upper band
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


class GhostTrader(qt.StrategyBase):
    def __init__(self,
            ma_short=3, ma_long=21, rsi_n = 9, rsi_oversold=30, rsi_overbought=70, donchian_n = 21
    ):
        super(GhostTrader, self).__init__()
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.rsi_n = rsi_n
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.donchian_n = donchian_n
        self.lookback = max(ma_long, rsi_n, donchian_n)
        self.long_ghost_virtual = False
        self.long_ghost_virtual_price = 0.0
        self.short_ghost_virtual = False
        self.short_ghost_virtual_price = 0.0
        self.current_time = None

    def on_tick(self, tick_event):
        self.current_time = tick_event.timestamp
        # print('Processing {}'.format(self.current_time))
        symbol = self.symbols[0]

        df_hist = self._data_board.get_hist_price(symbol, tick_event.timestamp)

        # wait for enough bars
        if df_hist.shape[0] < self.lookback:
            return

        current_price = df_hist.iloc[-1].Close
        current_size = self._position_manager.get_position_size(symbol)
        npv = self._position_manager.current_total_capital

        ema_short = talib.EMA(df_hist['Close'], self.ma_short).iloc[-1]
        ema_long = talib.EMA(df_hist['Close'], self.ma_long).iloc[-1]
        rsi = talib.RSI(df_hist['Close'], self.rsi_n).iloc[-1]
        long_stop = min(df_hist.Low.iloc[-self.donchian_n:])
        short_stop = max(df_hist.High.iloc[-self.donchian_n:])

        # fast ma > slow ma, rsi < 70, new high
        if current_size == 0 and ema_short > ema_long and rsi < self.rsi_overbought and \
                df_hist.High.iloc[-1] > df_hist.High.iloc[-2]:
            # ghost long
            if self.long_ghost_virtual == False:
                print('Ghost long, Pre-Price: %.2f, Long Price: %.2f' %
                         (df_hist['Close'].iloc[-2],
                          df_hist['Close'].iloc[-1]
                          ))
                self.long_ghost_virtual_price = df_hist['Close'].iloc[-1]
                self.long_ghost_virtual = True
                # actual long; after ghost loss
            if self.long_ghost_virtual == True and self.long_ghost_virtual_price > df_hist['Close'].iloc[-1]:
                self.long_ghost_virtual = False
                target_size = (int)(npv / current_price)
                self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
                print('BUY ORDER SENT, Pre-Price: %.2f, Price: %.2f, ghost price %.2f, Size: %.2f' %
                         (df_hist['Close'].iloc[-2],
                          df_hist['Close'].iloc[-1],
                          self.long_ghost_virtual_price,
                          target_size))
        # close long if below Donchian lower band
        elif current_size > 0 and df_hist['Close'].iloc[-1] <= long_stop:
            target_size = 0
            self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
            print('CLOSE LONG ORDER SENT, Pre-Price: %.2f, Price: %.2f, Low: %.2f, Stop: %.2f, Size: %.2f' %
                     (df_hist['Close'].iloc[-2],
                      df_hist['Close'].iloc[-1],
                      df_hist['Low'].iloc[-1],
                      long_stop,
                      target_size))

        # fast ma < slow ma, rsi > 30, new low
        if current_size == 0 and ema_short < ema_long and rsi > self.rsi_oversold and \
                df_hist['Low'].iloc[-1] < df_hist['Low'].iloc[-2]:
            # ghost short
            if self.short_ghost_virtual == False:
                print('Ghost short, Pre-Price: %.2f, Long Price: %.2f' %
                         (df_hist['Close'].iloc[-2],
                          df_hist['Close'].iloc[-1]
                          ))
                self.short_ghost_virtual_price = df_hist['Close'].iloc[-1]
                self.short_ghost_virtual = True
            # actual short; after ghost loss
            if self.short_ghost_virtual == True and self.short_ghost_virtual_price < df_hist['Close'].iloc[-1]:
                self.short_ghost_virtual = False
                target_size = -(int)(npv / current_price)
                self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
                print('SELL ORDER SENT, Pre-Price: %.2f, Price: %.2f, ghost price %.2f, Size: %.2f' %
                         (df_hist['Close'].iloc[-2],
                          df_hist['Close'].iloc[-1],
                          self.short_ghost_virtual_price,
                          target_size))
        # close short if above Donchian upper band
        elif current_size < 0 and df_hist['High'].iloc[-1] >= short_stop:
            target_size = 0
            self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
            print('CLOSE SHORT ORDER SENT, Pre-Price: %.2f, Price: %.2f, Low: %.2f, Stop: %.2f, Size: %.2f' %
                     (df_hist['Close'].iloc[-2],
                      df_hist['Close'].iloc[-1],
                      df_hist['High'].iloc[-1],
                      short_stop,
                      0))


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
        params_list = [{'ma_short': 3, 'ma_long': 21, 'rsi_n': 9, 'rsi_oversold': 30, 'rsi_overbought': 70, 'donchian_n': 21},
                       {'ma_short': 5, 'ma_long': 21, 'rsi_n': 9, 'rsi_oversold': 20, 'rsi_overbought': 80, 'donchian_n': 21}]
        target_name = 'Sharpe ratio'
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for params in params_list:
            strategy = GhostTrader()
            strategy.set_capital(init_capital)
            strategy.set_symbols([symbol])
            backtest_engine = qt.BacktestEngine(test_start_date, test_end_date)
            backtest_engine.set_capital(init_capital)  # capital or portfolio >= capital for one strategy
            backtest_engine.add_data(symbol, data)
            strategy.set_params({'ma_short': params['ma_short'], 'rsi_oversold': params['rsi_oversold'], 'rsi_overbought': params['rsi_overbought']})
            backtest_engine.set_strategy(strategy)
            tag = (params['ma_short'], params['rsi_oversold'])
            p = multiprocessing.Process(target=parameter_search, args=(backtest_engine, tag, target_name, return_dict))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
        for k,v in return_dict.items():
            print(k, v)
    else:
        strategy = GhostTrader()
        strategy.set_capital(init_capital)
        strategy.set_symbols([symbol])
        strategy.set_params(None)

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