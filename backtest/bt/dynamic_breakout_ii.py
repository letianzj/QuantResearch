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
from datetime import datetime
import backtrader as bt
from IPython.core.display import display, HTML
# set browser full width
display(HTML("<style>.container { width:100% !important; }</style>"))

class DynamicBreakoutII(bt.Strategy):
    params = (
        ('printlog', False),        # comma is required
    )

    def __init__(self):
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.bar_executed = None
        self.val_start = None
        self.lookback_days = 20

    def log(self, txt, dt=None, doprint=False):
        ''' Logging function fot this strategy'''
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def start(self):
        self.val_start = self.broker.get_cash()  # keep the starting cash

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:                # order.Partial
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Size: %.0f, Cost: %.2f, Comm %.2f, RemSize: %.0f, RemCash: %.2f' %
                    (order.executed.price,
                     order.executed.size,
                     order.executed.value,
                     order.executed.comm,
                     order.executed.remsize,
                     self.broker.get_cash()))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Size: %.0f, Cost: %.2f, Comm %.2f, RemSize: %.0f, RemCash: %.2f' %
                         (order.executed.price,
                          order.executed.size,
                          order.executed.value,
                          order.executed.comm,
                          order.executed.remsize,
                          self.broker.get_cash()))

            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Expired, order.Margin, order.Rejected]:
            self.log('Order Failed')

        self.order = None

    def next(self):
        # Simply log the closing price of the series from the reference
        # self.log('Close, %.2f' % self.data.close[0])
        if self.order:
            return

        # check adaptive lookback days
        if len(self.datas[0].close) < self.lookback_days+1:
            return
        else:
            today_vol = np.std(self.datas[0].close.get(0, self.lookback_days))
            yesterday_vol = np.std(self.datas[0].close.get(1, self.lookback_days))
            delta_vol = (today_vol / yesterday_vol) / today_vol
            self.lookback_days = round(self.lookback_days * (1+delta_vol), 0)
            self.lookback_days = int(min(max(self.lookback_days, 20), 60))
            if len(self.datas[0].close) < self.lookback_days:
                return

        # buy if close price > bollinger upper band and close price > Donchian HH
        # sell if close price < bollinger lower band and close price < Donchian LL
        current_price = self.datas[0].close[0]
        current_size = self.getposition(self.datas[0]).size
        hh = max(self.datas[0].high.get(1, self.lookback_days-1))
        ll = min(self.datas[0].low.get(1, self.lookback_days-1))
        ma = np.average(self.datas[0].close.get(0, self.lookback_days))
        sd = np.std(self.datas[0].close.get(0, self.lookback_days))
        ub = ma + 2.0*sd
        lb = ma - 2.0*sd

        if current_size == 0:
            target_size = int(self.broker.get_value() / current_price * 0.95)
            if current_price > ub: #and current_price > hh:
                self.order = self.order_target_size(target=target_size)
                self.log(f'LONG ORDER SENT, price: {current_price:.2f}, ub: {ub:.2f}, hh: {hh:.2f}, size: {target_size}')
            elif current_price < lb: #and current_price < ll:
                self.order = self.order_target_size(target=-target_size)
                self.log(f'SHORT ORDER SENT, price: {current_price:.2f}, lb: {lb:.2f}, ll: {ll:.2f}, size: {-target_size}')
        # exit long if price < MA; exit short if price > MA
        elif current_size > 0:
            if current_price < ma:
                self.order = self.order_target_size(target=0)
                self.log(f'FLAT LONG ORDER SENT, price: {current_price:.2f}, ma: {ma:.2f}, size: {-current_size}')
        else:
            if current_price > ma:
                self.order = self.order_target_size(target=0)
                self.log(f'FLAT SHORT ORDER SENT, price: {current_price:.2f}, ma: {ma:.2f}, size: {-current_size}')

    def stop(self):
        # calculate the actual returns
        print(self.analyzers)
        roi = (self.broker.get_value() / self.val_start) - 1.0
        self.log('ROI:        {:.2f}%'.format(100.0 * roi))
        self.log('(Dynamic Breakout Ending Value %.2f' %
                  self.broker.getvalue(), doprint=True)


if __name__ == '__main__':
    param_opt = False
    perf_eval = True
    benchmark = 'SPX'

    cerebro = bt.Cerebro()

    datapath = os.path.join('../data/', 'SPX.csv')

    # Create a Data Feed
    data = bt.feeds.YahooFinanceCSVData(
        dataname=datapath,
        fromdate=datetime(2010, 1, 1),
        todate=datetime(2019, 12, 31),
        reverse=False)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    # Add a FixedSize sizer according to the stake
    # cerebro.addsizer(bt.sizers.FixedSize, stake=10)
    # PercentSizer will flat position first; overwrite if not desired.
    # cerebro.addsizer(bt.sizers.PercentSizerInt, percents=95)

    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.001)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Add a strategy
    cerebro.addstrategy(DynamicBreakoutII, printlog=True)

    # Add Analyzer
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='SharpeRatio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='DrawDown')
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

    # Run over everything
    results = cerebro.run()

    # Print out the final result
    strat = results[0]
    print('Final Portfolio Value: %.2f, Sharpe Ratio: %.2f, DrawDown: %.2f, MoneyDown %.2f' %
          (cerebro.broker.getvalue(),
           strat.analyzers.SharpeRatio.get_analysis()['sharperatio'],
           strat.analyzers.DrawDown.get_analysis()['drawdown'],
           strat.analyzers.DrawDown.get_analysis()['moneydown']))

    if perf_eval:
        import matplotlib.pyplot as plt
        cerebro.plot(style='candlestick')
        plt.show()

        pyfoliozer = strat.analyzers.getbyname('pyfolio')
        returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
        print('-------------- RETURNS ----------------')
        print(returns)
        print('-------------- POSITIONS ----------------')
        print(positions)
        print('-------------- TRANSACTIONS ----------------')
        print(transactions)
        print('-------------- GROSS LEVERAGE ----------------')
        print(gross_lev)

        import empyrical as ep
        import pyfolio as pf

        bm_ret = None
        if benchmark:
            datapath = os.path.join('../data/', f'{benchmark}.csv')
            bm = pd.read_csv(datapath, index_col=0)
            bm_ret = bm['Adj Close'].pct_change().dropna()
            bm_ret.index = pd.to_datetime(bm_ret.index)
            # remove tzinfo
            returns.index = returns.index.tz_localize(None)
            bm_ret = bm_ret[returns.index]
            bm_ret.name = 'benchmark'

        perf_stats_strat = pf.timeseries.perf_stats(returns)
        perf_stats_all = perf_stats_strat
        if benchmark:
            perf_stats_bm = pf.timeseries.perf_stats(bm_ret)
            perf_stats_all = pd.concat([perf_stats_strat, perf_stats_bm], axis=1)
            perf_stats_all.columns = ['Strategy', 'Benchmark']

        drawdown_table = pf.timeseries.gen_drawdown_table(returns, 5)
        monthly_ret_table = ep.aggregate_returns(returns, 'monthly')
        monthly_ret_table = monthly_ret_table.unstack().round(3)
        ann_ret_df = pd.DataFrame(ep.aggregate_returns(returns, 'yearly'))
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
            returns,
            benchmark_rets=bm_ret if benchmark else None,
            positions=positions,
            transactions=transactions,
            #live_start_date='2005-05-01',
            round_trips=False)
        plt.show()