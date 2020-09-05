'''
    https://programming.vip/docs/r-breaker-strategy-for-commodity-futures.html
    https://github.com/myquant/strategy/blob/master/R-Breaker/info.md
    The R-Breaker strategy was developed by Richard Saidenberg and published in 1994.
    After that, it was ranked one of the top 10 most profitable trading strategies
    by Futures Truth magazine in the United States for 15 consecutive years.
    Simply put, the R-Breaker strategy is a support and resistance level strategy,
    which calculates seven prices based on yesterday's highest, lowest and closing prices

    The R - Breaker strategy draws grid - like price lines based on yesterday's prices and updates them once a day.
    The support position and resistance position in technical analysis, and their roles can be converted to each other.
    When the price successfully breaks up the resistance level, the resistance level becomes the support level;
    when the price successfully breaks down the support level, the support level becomes the resistance level.

    In the Forex trading system, the Pivot Points trading method is a classic trading strategy.
    Pivot Points is a very simple resistance support system.
    Based on yesterday's highest, lowest and closing prices, seven price points are calculated,
    including one pivot point, three resistance levels and three support levels.

    It is a day trading strategy, generally not overnight.
    Here I'm using close price, performance is expected to deteriorate.
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


class RBreaker(qt.StrategyBase):
    def __init__(self):
        super(RBreaker, self).__init__()
        self.price_entered = 0.0
        self.stop_loss_price = 10
        self.current_time = None

    def on_tick(self, tick_event):
        self.current_time = tick_event.timestamp
        # print('Processing {}'.format(self.current_time))
        symbol = self.symbols[0]

        df_hist = self._data_board.get_hist_price(symbol, tick_event.timestamp)

        # need yesterday's prices
        if df_hist.shape[0] <= 1:
            return

        yesterday_open = df_hist.iloc[-2].Open
        yesterday_high = df_hist.iloc[-2].High
        yesterday_low = df_hist.iloc[-2].Low
        yesterday_close = df_hist.iloc[-2].Close

        # center or middle price
        pivot = (yesterday_high + yesterday_close + yesterday_low) / 3  # pivot point
        # r3 > r2 > r1
        r1 = 2 * pivot - yesterday_low  # Resistance Level 1; Reverse Selling price
        r2 = pivot + (yesterday_high - yesterday_low)  # Resistance Level 2; setup; Observed Sell Price
        r3 = yesterday_high + 2 * (pivot - yesterday_low)  # Resistance Level 3; break through buy
        # s1 > s2 > s3
        s1 = 2 * pivot - yesterday_high  # Support 1; reverse buying
        s2 = pivot - (yesterday_high - yesterday_low)  # Support Position 2; setup; Observed Buy Price
        s3 = yesterday_low - 2 * (yesterday_high - pivot)  # Support 3; break through sell

        today_high = df_hist.iloc[-1].High
        today_low = df_hist.iloc[-1].Low
        current_price = df_hist.iloc[-1].Close
        current_size = self._position_manager.get_position_size(symbol)
        npv = self._position_manager.current_total_capital

        if current_size == 0:       # If no position
            if current_price > r3:          # If the current price breaks through resistance level 3/highest, go long
                target_size = int(npv / current_price)
                self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
                print(f'BUY ORDER SENT, price: {current_price:.2f}, r1 {r1:.2f}, r2 {r2:.2f} r3 {r3:.2f}, s1 {s1:.2f}  s2 {s2:.2f}, s3 {s3:.2f} size: {target_size}')
            if current_price < s3:  # If the current price break-through support level 3/lowest, go short
                target_size = -int(npv / current_price)
                self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
                print(f'SELL ORDER SENT, price: {current_price:.2f}, r1 {r1:.2f}, r2 {r2:.2f} r3 {r3:.2f}, s1 {s1:.2f}  s2 {s2:.2f}, s3 {s3:.2f} size: {target_size}')
        elif current_size  > 0:
            if (today_high > r2 and current_price < r1) or current_price < s3:  # price reverses. flip from long to short
                target_size = -int(npv / current_price)
                self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
                print(f'FLIP TO SHORT ORDER SENT, price: {current_price:.2f}, r1 {r1:.2f}, r2 {r2:.2f} r3 {r3:.2f}, s1 {s1:.2f}  s2 {s2:.2f}, s3 {s3:.2f} size: {target_size}')
        elif current_size  < 0:
            if (today_low < s2 and current_price > s1) or current_price > r3:   # price reverses, flip from short to long
                target_size = int(npv / current_price * 0.95)
                self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
                print(f'FLIP TO LONG ORDER SENT, price: {current_price:.2f}, r1 {r1:.2f}, r2 {r2:.2f} r3 {r3:.2f}, s1 {s1:.2f}  s2 {s2:.2f}, s3 {s3:.2f} size: {target_size}')


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
        pass
    else:
        strategy = RBreaker()
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