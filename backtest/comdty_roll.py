'''
Comdty roll according to roll schedule
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
import futures_tools
import data_loader
# set browser full width
from IPython.core.display import display, HTML
pd.set_option('display.max_columns', None)
display(HTML("<style>.container { width:100% !important; }</style>"))


class ComdtyMonthlyRoll(qt.StrategyBase):
    def __init__(self,
            n_roll_ahead=0,            # 0 is last day roll, 1 is penultimate day, and so on
            n_rollout=0,         # 0 is front month, 1 is second month, and so on
    ):
        super(ComdtyMonthlyRoll, self).__init__()
        self.n_roll_ahead = n_roll_ahead
        self.n_rollout= n_rollout
        self.sym = 'CL'
        self.current_time = None
        self.df_meta = data_loader.load_futures_meta(self.sym)
        self.holding_contract = None

    def on_tick(self, tick_event):
        """
        front_contract decides when to roll
        if not roll ==> if no holding_contract, buy rollout_contract; else do nothing
        if roll ==> if no holding_contract, buy rollin contract (rollout+1); else sell rollout, buy rollin contract
        """
        super().on_tick(tick_event)
        self.current_time = tick_event.timestamp
        #symbol = self.symbols[0]
        #df_hist = self._data_board.get_hist_price(symbol, tick_event.timestamp)
        df_time_idx = self._data_board.get_hist_time_index()

        df_live_futures = futures_tools.get_futures_chain(meta_data = self.df_meta, asofdate = self.current_time.replace(tzinfo=None))     # remove tzinfo
        # front_contract = df_live_futures.index[0] 
        rollout_contract = df_live_futures.index[self.n_rollout]
        rollin_contract = df_live_futures.index[self.n_rollout+1]
        exp_date = pytz.timezone('US/Eastern').localize(df_live_futures.Last_Trade_Date[0])       # front contract
        dte = df_time_idx.searchsorted(exp_date) - df_time_idx.searchsorted(self.current_time)       # 0 is expiry date

        if self.n_roll_ahead < dte:         # not ready to roll
            if self.holding_contract is None:      # empty
                print(f'{self.current_time}, dte {dte}, buy {rollout_contract}')
                self.adjust_position(rollout_contract, size_from=0, size_to=1, timestamp=self.current_time)
                self.holding_contract = rollout_contract
        else:
            if self.holding_contract is None:      # empty
                print(f'{self.current_time}, dte {dte}, buy {rollin_contract}')
                self.adjust_position(rollin_contract, size_from=0, size_to=1, timestamp=self.current_time)
                self.holding_contract = rollin_contract
            else:
                if self.holding_contract == rollin_contract:     # already rolled this month
                    pass
                else:
                    print(f'{self.current_time}, dte {dte}, roll {rollout_contract} {rollin_contract}')
                    self.adjust_position(rollout_contract, size_from=1, size_to=0, timestamp=self.current_time)
                    self.adjust_position(rollin_contract, size_from=0, size_to=1, timestamp=self.current_time)
                    self.holding_contract = rollin_contract


def parameter_search(symbol, init_capital, sd, ed, df_data, params, target_name, return_dict):
    """
    This function should be the same for all strategies.
    The only reason not included in quanttrader is because of its dependency on pyfolio (to get perf_stats)
    """
    strategy = ComdtyMonthlyRoll()
    strategy.set_capital(init_capital)
    strategy.set_symbols([symbol])
    engine = qt.BacktestEngine(sd, ed)
    engine.set_capital(init_capital)  # capital or portfolio >= capital for one strategy
    engine.add_data(symbol, df_data)
    strategy.set_params({'n_roll_ahead': params['n_roll_ahead'], 'n_rollout': params['n_rollout']})
    engine.set_strategy(strategy)
    ds_equity, _, _ = engine.run()
    try:
        strat_ret = ds_equity.pct_change().dropna()
        perf_stats_strat = pf.timeseries.perf_stats(strat_ret)
        target_value = perf_stats_strat.loc[target_name]  # first table in tuple
    except KeyError:
        target_value = 0
    return_dict[(params['n_roll_ahead'], params['n_rollout'])] = target_value


if __name__ == '__main__':
    do_optimize = False
    run_in_jupyter = False
    symbol = 'CL'
    benchmark = None
    init_capital = 100_000.0
    df_future = data_loader.load_futures_hist_prices(symbol)
    df_future.index = df_future.index.tz_localize('US/Eastern')
    test_start_date = datetime(2019, 1, 1, 0, 0, 0, 0, pytz.timezone('US/Eastern'))
    test_end_date = datetime(2021, 12, 30, 0, 0, 0, 0, pytz.timezone('US/Eastern'))
    
    init_capital = 50.0
    if do_optimize:          # parallel parameter search
        params_list = []
        for n_roll_ahead in range(20):
            for n_rollout in range(5):
                params_list.append({'n_roll_ahead': n_roll_ahead, 'n_rollout': n_rollout})
        target_name = 'Sharpe ratio'
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for params in params_list:
            p = multiprocessing.Process(target=parameter_search, args=(symbol, init_capital, test_start_date, test_end_date, df_future, params, target_name, return_dict))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
        for k,v in return_dict.items():
            print(k, v)
    else:
        strategy = ComdtyMonthlyRoll()
        strategy.set_capital(init_capital)
        strategy.set_symbols([symbol])
        strategy.set_params({'n_roll_ahead': 0, 'n_rollout': 0})

        # Create a Data Feed
        backtest_engine = qt.BacktestEngine(test_start_date, test_end_date)
        backtest_engine.set_capital(init_capital)  # capital or portfolio >= capital for one strategy
        backtest_engine.add_data(symbol, df_future)
        backtest_engine.set_strategy(strategy)
        ds_equity, df_positions, df_trades = backtest_engine.run()
        # save to excel
        qt.util.save_one_run_results('./output', ds_equity, df_positions, df_trades)

        # ------------------------- Evaluation and Plotting -------------------------------------- #
        strat_ret = ds_equity.pct_change().dropna()
        strat_ret.name = 'strat'
        # bm = qt.util.read_ohlcv_csv(os.path.join('../data/', f'{benchmark}.csv'))
        # bm_ret = bm['Close'].pct_change().dropna()
        # bm_ret.index = pd.to_datetime(bm_ret.index)
        # bm_ret = bm_ret[strat_ret.index]
        # bm_ret.name = 'benchmark'
        bm_ret = strat_ret.copy()
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