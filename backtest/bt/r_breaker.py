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
from datetime import datetime
import backtrader as bt
from IPython.core.display import display, HTML
# set browser full width
display(HTML("<style>.container { width:100% !important; }</style>"))

class RBreaker(bt.Strategy):
    params = (
        ('printlog', False),        # comma is required
    )

    def __init__(self):
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.bar_executed = None
        self.val_start = None
        self.price_entered = 0.0
        self.stop_loss_price = 10

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

        # need yesterday's prices
        if len(self.datas[0].close) <= 1:
            return

        yesterday_open = self.datas[0].open[-1]
        yesterday_high = self.datas[0].high[-1]
        yesterday_low = self.datas[0].low[-1]
        yesterday_close = self.datas[0].close[-1]

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

        today_high = self.datas[0].high[0]  # Day High Price
        today_low = self.datas[0].low[0]  # Today's Lowest Price
        current_price = self.datas[0].close[0] # Current price

        # if diff between price entered and current price > stop loss trigger, stop loss
        # if (self.current_position > 0 and self.price_entered - current_price >= self.STOP_LOSS_PRICE) or \
        #         (self.current_position < 0 and current_price - self.price_entered >= self.STOP_LOSS_PRICE):
        #     target_size = 0
        #     self.order = self.order_target_size(target=target_size)
        #     self.log(f'STOP LOSS ORDER SENT, price: {current_price:.2f}, r1 {r1:.2f}, r2 {r2:.2f} r3 {r3:.2f}, s1 {s1:.2f}  s2 {s2:.2f}, s3 {s3:.2f} size: {target_size}')


        if self.position.size == 0:       # If no position
            if current_price > r3:          # If the current price breaks through resistance level 3/highest, go long
                target_size = int(self.broker.get_value() / current_price * 0.95)
                self.order = self.order_target_size(target=target_size)
                self.log(f'BUY ORDER SENT, price: {current_price:.2f}, r1 {r1:.2f}, r2 {r2:.2f} r3 {r3:.2f}, s1 {s1:.2f}  s2 {s2:.2f}, s3 {s3:.2f} size: {target_size}')
            if current_price < s3:  # If the current price break-through support level 3/lowest, go short
                target_size = -int(self.broker.get_value() / current_price * 0.95)
                self.order = self.order_target_size(target=target_size)
                self.log(f'SELL ORDER SENT, price: {current_price:.2f}, r1 {r1:.2f}, r2 {r2:.2f} r3 {r3:.2f}, s1 {s1:.2f}  s2 {s2:.2f}, s3 {s3:.2f} size: {target_size}')
        elif self.position.size  > 0:
            if (today_high > r2 and current_price < r1) or current_price < s3:  # price reverses. flip from long to short
                target_size = -int(self.broker.get_value() / current_price * 0.95)
                self.order = self.order_target_size(target=target_size)
                self.log(f'FLIP TO SHORT ORDER SENT, price: {current_price:.2f}, r1 {r1:.2f}, r2 {r2:.2f} r3 {r3:.2f}, s1 {s1:.2f}  s2 {s2:.2f}, s3 {s3:.2f} size: {target_size}')
        elif self.position.size  < 0:
            if (today_low < s2 and current_price > s1) or current_price > r3:   # price reverses, flip from short to long
                target_size = int(self.broker.get_value() / current_price * 0.95)
                self.order = self.order_target_size(target=target_size)
                self.log(f'FLIP TO LONG ORDER SENT, price: {current_price:.2f}, r1 {r1:.2f}, r2 {r2:.2f} r3 {r3:.2f}, s1 {s1:.2f}  s2 {s2:.2f}, s3 {s3:.2f} size: {target_size}')

    def stop(self):
        # calculate the actual returns
        print(self.analyzers)
        roi = (self.broker.get_value() / self.val_start) - 1.0
        self.log('ROI:        {:.2f}%'.format(100.0 * roi))
        self.log('(Ghost Trader params Ending Value %.2f' %
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
    cerebro.addstrategy(RBreaker, printlog=True)

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