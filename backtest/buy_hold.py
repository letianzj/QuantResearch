'''
buy hold
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
# set browser full width
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


class BuyAndHoldStrategy(qt.StrategyBase):
    """
    buy on the first tick then hold to the end
    """

    def __init__(self):
        super(BuyAndHoldStrategy, self).__init__()
        self.invested = False

    def on_tick(self, event):
        print(event.timestamp)
        symbol = self.symbols[0]
        if not self.invested:
            df_hist = self._data_board.get_hist_price(symbol, event.timestamp)
            close = df_hist.iloc[-1].Close
            target_size = int(self._position_manager.cash / close)
            self.adjust_position(symbol, size_from=0, size_to=target_size, timestamp=event.timestamp)
            self.invested = True


if __name__ == '__main__':
    run_in_jupyter = False
    symbol = 'SPX'
    benchmark = 'SPX'
    datapath = os.path.join('../data/', f'{symbol}.csv')
    data = qt.util.read_ohlcv_csv(datapath)
    init_capital = 100_000.0
    test_start_date = datetime(2010,1,1, 8, 30, 0, 0, pytz.timezone('America/New_York'))
    test_end_date = datetime(2019,12,31, 6, 0, 0, 0, pytz.timezone('America/New_York'))

    strategy = BuyAndHoldStrategy()
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
        f8 = plt.figure(8)
        pf.create_position_tear_sheet(strat_ret, df_positions)
        plt.show()
        f9 = plt.figure(9)
        pf.create_txn_tear_sheet(strat_ret, df_positions, df_trades)
        plt.show()
        f10 = plt.figure(10)
        pf.create_round_trip_tear_sheet(strat_ret, df_positions, df_trades)
        plt.show()