'''
Buy/sell when price crosses above/below SMA;
Close position when price crosses below/above SMA;
Negative return and sharp ratio.
'''
import os
import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timezone
import quanttrader as qt
import matplotlib.pyplot as plt
import empyrical as ep
import pyfolio as pf
# set browser full width
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


class MebaneFaberTAA(qt.StrategyBase):
    def __init__(self,
            nfast=20, nslow=200
    ):
        super(MebaneFaberTAA, self).__init__()
        self.nfast = nfast
        self.nslow = nslow
        self.current_time = None

    def on_tick(self, tick_event):
        self.current_time = tick_event.timestamp
        # print('Processing {}'.format(self.current_time))

        # wait for enough bars
        for symbol in self.symbols:
            df_hist = self._data_board.get_hist_price(symbol, self.current_time)
            if df_hist.shape[0] < self.nslow:
                return

        # wait for month end
        time_index = self._data_board.get_hist_time_index()
        time_loc = time_index.get_loc(self.current_time)
        if (time_loc != len(time_index)-1) & (time_index[time_loc].month == time_index[time_loc+1].month):
            return

        npv = self._position_manager.current_total_capital
        stock_value = npv / 5
        for symbol in self.symbols:
            current_size = self._position_manager.get_position_size(symbol)
            current_price = self._data_board.get_hist_price(symbol, self.current_time)['Close'].iloc[-1]
            ma_fast = np.mean(self._data_board.get_hist_price(symbol, self.current_time)['Close'][-self.nfast:])
            ma_slow = np.mean(self._data_board.get_hist_price(symbol, self.current_time)['Close'][-self.nslow:])
            if ma_fast > ma_slow:       # buy
                target_size = (int)(stock_value / current_price)
                self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
                print(f'{self.current_time}, LONG ORDER SENT, {symbol}, Price: {current_price:.2f}, '
                      f'fast sma: {ma_fast:.2f}, slow sma: {ma_slow:.2f}, Size: {target_size}')
            else:  # hold cash
                target_size = 0
                self.adjust_position(symbol, size_from=current_size, size_to=target_size, timestamp=self.current_time)
                print(f'{self.current_time}, FLAT ORDER SENT, {symbol}, Price: {current_price:.2f}, '
                      f'fast sma: {ma_fast:.2f}, slow sma: {ma_slow:.2f}, Size: {target_size}')


if __name__ == '__main__':
    etfs = ['SPY', 'EFA', 'TIP', 'GSG', 'VNQ']
    benchmark = 'SPX'
    init_capital = 100_000.0
    test_start_date = datetime(2010,1,1, 8, 30, 0, 0, pytz.timezone('America/New_York'))
    test_end_date = datetime(2019,12,31, 6, 0, 0, 0, pytz.timezone('America/New_York'))

    strategy = MebaneFaberTAA()
    strategy.set_capital(init_capital)
    strategy.set_symbols(etfs)
    strategy.set_params({'nfast': 20, 'nslow': 200})

    backtest_engine = qt.BacktestEngine(test_start_date, test_end_date)
    backtest_engine.set_capital(init_capital)  # capital or portfolio >= capital for one strategy

    for symbol in etfs:
        data = qt.util.read_ohlcv_csv(os.path.join('../data/', f'{symbol}.csv'))
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

    pf.create_full_tear_sheet(
        strat_ret,
        benchmark_rets=bm_ret if benchmark else None,
        positions=df_positions,
        transactions=df_trades,
        round_trips=False)
    plt.show()

    # if not run in jupyter
    pf.plot_rolling_returns(strat_ret, factor_returns=bm_ret)
    plt.show()
    pf.plot_rolling_volatility(strat_ret, factor_returns=bm_ret)
    plt.show()
    pf.plot_rolling_sharpe(strat_ret)
    plt.show()
    pf.plot_drawdown_periods(strat_ret)
    plt.show()
    pf.plot_monthly_returns_heatmap(strat_ret)
    plt.show()
    pf.plot_annual_returns(strat_ret)
    plt.show()
    pf.plot_monthly_returns_dist(strat_ret)
    plt.show()
    pf.create_position_tear_sheet(strat_ret, df_positions)
    plt.show()
    pf.create_txn_tear_sheet(strat_ret, df_positions, df_trades)
    plt.show()
    pf.create_round_trip_tear_sheet(strat_ret, df_positions, df_trades)
    plt.show()