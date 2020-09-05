'''
golden cross buy; dead cross sell
'''
import os
import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timezone
import multiprocessing
import quanttrader as qt
import matplotlib.pyplot as plt
import empyrical as ep
import pyfolio as pf
import pickle
# set browser full width
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


class MADoubleCross(qt.StrategyBase):
    def __init__(self,
            short_window=50, long_window=200
    ):
        super(MADoubleCross, self).__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.current_time = None
        self.current_position = 0

    def on_tick(self, tick_event):
        self.current_time = tick_event.timestamp
        # print('Processing {}'.format(self.current_time))
        symbol = self.symbols[0]

        df_hist = self._data_board.get_hist_price(symbol, tick_event.timestamp)
        current_price = df_hist.iloc[-1].Close

        # wait for enough bars
        if df_hist.shape[0] < self.long_window:
            return

        # Calculate the simple moving averages
        short_sma = np.mean(df_hist['Close'][-self.short_window:])
        long_sma = np.mean(df_hist['Close'][-self.long_window:])
        # Trading signals based on moving average cross
        if short_sma > long_sma and self.current_position <= 0:
            target_size = int((self._position_manager.cash + self.current_position * df_hist['Close'].iloc[-1])/df_hist['Close'].iloc[-1])       # buy to notional
            self.adjust_position(symbol, size_from=self.current_position, size_to=target_size, timestamp=self.current_time)
            print("Long: %s, short_sma %s, long_sma %s, price %s, trade %s, new position %s" % (self.current_time, str(short_sma), str(long_sma), str(current_price), str(target_size-self.current_position), str(target_size)))
            self.current_position = target_size
        elif short_sma < long_sma and self.current_position >= 0:
            target_size = int((self._position_manager.cash + self.current_position * df_hist['Close'].iloc[-1])/df_hist['Close'].iloc[-1])*(-1)    # sell to notional
            self.adjust_position(symbol, size_from=self.current_position, size_to=target_size, timestamp=self.current_time)
            print("Short: %s, short_sma %s, long_sma %s, price %s, trade %s, new position %s" % (self.current_time, str(short_sma), str(long_sma), str(current_price), str(target_size-self.current_position), str(target_size)))
            self.current_position = target_size


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
    is_intraday = True
    symbol = 'SPX'
    benchmark = 'SPX'
    init_capital = 100_000.0

    if not is_intraday:
        test_start_date = datetime(2010, 1, 1, 8, 30, 0, 0, pytz.timezone('America/New_York'))
        test_end_date = datetime(2019, 12, 31, 6, 0, 0, 0, pytz.timezone('America/New_York'))
        datapath = os.path.join('../data/', f'{symbol}.csv')
        data = qt.util.read_ohlcv_csv(datapath)
    else:
        # it seems initialize timezone doesn't work
        eastern = pytz.timezone('US/Eastern')
        test_start_date = eastern.localize(datetime(2020, 8, 10, 9, 30, 0))
        test_end_date = eastern.localize(datetime(2020, 8, 10, 10, 0, 0))
        dict_hist_data = {}
        if os.path.isfile('../data/tick/20200810.pkl'):
            with open('../data/tick/20200810.pkl', 'rb') as f:
                dict_hist_data = pickle.load(f)
        data = dict_hist_data['ESU0 FUT GLOBEX']
        data.index = data.index.tz_localize('America/New_York')  # US/Eastern, UTC

    if do_optimize:          # parallel parameter search
        params_list = []
        for sw in [10, 20, 30, 50, 100, 200]:
            for lw in [10, 20, 30, 50, 100, 200]:
                if lw <= sw:
                    continue
                params_list.append({'short_window': sw, 'long_window': lw})
        target_name = 'Sharpe ratio'
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for params in params_list:
            strategy = MADoubleCross()
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
        strategy = MADoubleCross()
        strategy.set_capital(init_capital)
        strategy.set_symbols([symbol])
        strategy.set_params({'short_window':20, 'long_window':200})

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
        if not is_intraday:
            bm = qt.util.read_ohlcv_csv(os.path.join('../data/', f'{benchmark}.csv'))
        else:
            bm = data      # buy and hold
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
            pf.plotting.show_perf_stats(
                strat_ret, bm_ret,
                positions=df_positions,
                transactions=df_trades)
            pf.plotting.show_worst_drawdown_periods(strat_ret)

            pf.plot_perf_stats(strat_ret, bm_ret)
            plt.show()
            pf.plotting.plot_returns(strat_ret)
            plt.show()
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
            pf.create_interesting_times_tear_sheet(strat_ret, benchmark_rets=bm_ret)
            plt.show()
            f9 = plt.figure(9)
            pf.create_position_tear_sheet(strat_ret, df_positions)
            # plt.show()
            f10 = plt.figure(10)
            pf.create_txn_tear_sheet(strat_ret, df_positions, df_trades)
            # plt.show()
            f11 = plt.figure(11)
            pf.create_round_trip_tear_sheet(strat_ret, df_positions, df_trades)
            plt.show()