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
from datetime import datetime
import backtrader as bt
from IPython.core.display import display, HTML
# set browser full width
display(HTML("<style>.container { width:100% !important; }</style>"))

class Turtle(bt.Strategy):
    params = (
        ('long_window', 20),
        ('short_window', 10),
        ('printlog', False),        # comma is required
    )

    def __init__(self):
        self.order = None
        self.buyprice = 0.0
        self.buycomm = 0.0
        self.bar_executed = 0
        self.val_start = 0.0

        self.buy_count = 0
        self.don_high = bt.indicators.Highest(self.data.high(-1), period=self.params.long_window)
        self.don_low = bt.indicators.Lowest(self.data.low(-1), period=self.params.short_window)
        # https://en.wikipedia.org/wiki/Average_true_range
        self.TR = bt.indicators.Max((self.data.high(0) - self.data.low(0)), \
                                    abs(self.data.close(-1) - self.data.high(0)), \
                                    abs(self.data.close(-1) - self.data.low(0)))
        self.ATR = bt.indicators.SimpleMovingAverage(self.TR, period=14)

        self.buy_signal = bt.ind.CrossOver(self.data.close(0), self.don_high)
        self.sell_signal = bt.ind.CrossOver(self.data.close(0), self.don_low)

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

        # Long
        if self.buy_signal > 0 and self.buy_count == 0:
            # one unit is 1% of total risk asset
            target_size = int(self.broker.getvalue() * 0.01 / self.ATR[0])
            self.order = self.order_target_size(target=target_size)
            self.log(f'LONG ORDER SENT, price: {self.data.close[0]:.2f}, don_high: {self.don_high[0]:.2f}')
            self.buy_count = 1
        # add; This is for futures; may go beyond notional; leverage is set to 4
        elif self.data.close > self.buyprice + 0.5 * self.ATR[0] and self.buy_count > 0 and self.buy_count <= 3:
            target_size = int(self.broker.getvalue() * 0.01 / self.ATR[0])
            target_size += self.getposition(self.datas[0]).size        # on top of current size
            self.order = self.order_target_size(target=target_size)
            self.log(f'ADD LONG ORDER SENT, add time: {self.buy_count}, price: {self.data.close[0]:.2f}, don_high: {self.don_high[0]:.2f}')
            self.buy_count += 1
        # flat
        elif self.sell_signal < 0 and self.buy_count > 0:
            self.order = self.order_target_size(target=0)
            self.log(f'FLAT ORDER SENT, price: {self.data.close[0]:.2f}, don_low: {self.don_low[0]:.2f}')
            self.buy_count = 0
        # flat, stop loss
        elif self.data.close < (self.buyprice - 2 * self.ATR[0]) and self.buy_count > 0:
            self.order = self.order_target_size(target=0)
            self.log(f'FLAT ORDER SENT, price: {self.data.close[0]:.2f}, {self.buyprice:.2f}, 2ATR: {2 * self.ATR[0]:.2f}')
            self.buy_count = 0

    def stop(self):
        # calculate the actual returns
        print(self.analyzers)
        roi = (self.broker.get_value() / self.val_start) - 1.0
        self.log('ROI:        {:.2f}%'.format(100.0 * roi))
        self.log('(Turtle Ending Value %.2f' %
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
    cerebro.broker.setcommission(commission=0.001, leverage=10)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Add a strategy
    cerebro.addstrategy(Turtle, printlog=True)

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