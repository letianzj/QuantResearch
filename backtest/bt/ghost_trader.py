'''
It is observed that if last trade is profitable, next trade would more likely be a loss.
Then why not create a ghost trader on the same strategy; and trade only when the ghost trader's a loss.
Elements: two moving averages; rsi; donchain channel
conditions: 1. long if short MA > long MA, rsi lower than overbought 70, new high
            2. short if short MA < long MA, ris higher than oversold 30, new low
exit:       1. exit long if lower than donchian lower band
            2. exit short if higher than donchian upper band
-42% vs benchmark 123%
'''
import os
import numpy as np
import pandas as pd
from datetime import datetime
import backtrader as bt
from IPython.core.display import display, HTML
# set browser full width
display(HTML("<style>.container { width:100% !important; }</style>"))

class GhostTrader(bt.Strategy):
    params = (
        ('ma_short', 3),
        ('ma_long', 21),
        ('rsi_n', 9),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('donchian_n', 21),
        ('printlog', False),        # comma is required
    )

    def __init__(self):
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.bar_executed = None
        self.val_start = None
        self.long_ghost_virtual = False
        self.long_ghost_virtual_price = 0.0
        self.short_ghost_virtual = False
        self.short_ghost_virtual_price = 0.0
        self.dataclose = self.datas[0].close
        self.ema_short = bt.indicators.ExponentialMovingAverage(self.dataclose, period=self.params.ma_short)
        self.ema_long = bt.indicators.ExponentialMovingAverage(self.dataclose, period=self.params.ma_long)
        self.rsi = bt.indicators.RelativeStrengthIndex(self.dataclose, period=self.params.rsi_n)
        # make sure donchian n is respected
        self.dummy = bt.indicators.Momentum(self.dataclose, period=self.params.donchian_n-1)

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

        ema_short = self.ema_short[0]
        ema_long = self.ema_long[0]
        rsi = self.rsi[0]
        long_stop = min(self.datas[0].low.get(0, self.params.donchian_n))
        short_stop = max(self.datas[0].high.get(0, self.params.donchian_n))

        # fast ma > slow ma, rsi < 70, new high
        if self.position.size == 0 and ema_short > ema_long and rsi < self.params.rsi_overbought and \
                self.datas[0].high[0] > self.datas[0].high[-1]:
            # ghost long
            if self.long_ghost_virtual == False:
                self.log('Ghost long, Pre-Price: %.2f, Long Price: %.2f' %
                         (self.dataclose[-1],
                          self.dataclose[0]
                          ))
                self.long_ghost_virtual_price = self.datas[0].close[0]
                self.long_ghost_virtual = True
            # actual long; after ghost loss
            if self.long_ghost_virtual == True and self.long_ghost_virtual_price > self.datas[0].close[0]:
                self.long_ghost_virtual = False
                self.order = self.buy()
                self.log('BUY ORDER SENT, Pre-Price: %.2f, Price: %.2f, ghost price %.2f, Size: %.2f' %
                         (self.dataclose[-1],
                          self.dataclose[0],
                          self.long_ghost_virtual_price,
                          self.getsizing(isbuy=True)))
        # close long if below Donchian lower band
        elif self.position.size > 0 and self.datas[0].low[0] <= long_stop:
            self.order = self.sell()
            self.log('CLOSE LONG ORDER SENT, Pre-Price: %.2f, Price: %.2f, Low: %.2f, Stop: %.2f, Size: %.2f' %
                     (self.dataclose[-1],
                      self.dataclose[0],
                      self.datas[0].low[0],
                      long_stop,
                      self.getsizing(isbuy=False)))

        # fast ma < slow ma, rsi > 30, new low
        if self.position.size == 0 and ema_short < ema_long and rsi > self.params.rsi_oversold and \
                self.datas[0].low[0] < self.datas[0].low[-1]:
            # ghost short
            if self.short_ghost_virtual == False:
                self.log('Ghost short, Pre-Price: %.2f, Long Price: %.2f' %
                         (self.dataclose[-1],
                          self.dataclose[0]
                          ))
                self.short_ghost_virtual_price = self.datas[0].close[0]
                self.short_ghost_virtual = True
            # actual short; after ghost loss
            if self.short_ghost_virtual == True and self.short_ghost_virtual_price < self.datas[0].close[0]:
                self.short_ghost_virtual = False
                self.order = self.sell()
                self.log('SELL ORDER SENT, Pre-Price: %.2f, Price: %.2f, ghost price %.2f, Size: %.2f' %
                         (self.dataclose[-1],
                          self.dataclose[0],
                          self.short_ghost_virtual_price,
                          self.getsizing(isbuy=False)))
        # close short if above Donchian upper band
        elif self.position.size < 0 and self.datas[0].high[0] >= short_stop:
            self.order = self.buy()
            self.log('CLOSE SHORT ORDER SENT, Pre-Price: %.2f, Price: %.2f, Low: %.2f, Stop: %.2f, Size: %.2f' %
                     (self.dataclose[-1],
                      self.dataclose[0],
                      self.datas[0].high[0],
                      short_stop,
                      self.getsizing(isbuy=True)))

    def stop(self):
        # calculate the actual returns
        print(self.analyzers)
        roi = (self.broker.get_value() / self.val_start) - 1.0
        self.log('ROI:        {:.2f}%'.format(100.0 * roi))
        self.log('(Ghost Trader params (%2d, %2d, %2d, %2d, %2d, %2d)) Ending Value %.2f' %
                 (self.params.ma_short, self.params.ma_long, self.params.rsi_n,
                  self.params.rsi_oversold, self.params.rsi_overbought, self.params.donchian_n,
                  self.broker.getvalue()), doprint=True)


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
    cerebro.addsizer(bt.sizers.PercentSizerInt, percents=95)

    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.001)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Add a strategy
    if param_opt:
        # Optimization
        cerebro.optstrategy(GhostTrader, donchian_n=[15, 20, 25])
        perf_eval = False
    else:
        cerebro.addstrategy(GhostTrader, printlog=True)

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