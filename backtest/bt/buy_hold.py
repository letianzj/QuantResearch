import os
import pandas as pd
from datetime import datetime
import backtrader as bt

class BuyAndHold(bt.Strategy):
    def __init__(self):
        # To keep track of pending orders and buy price/commission
        self.order = None

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
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
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

    def next(self):
        # Simply log the closing price of the series from the reference
        # self.log('Close, %.2f' % self.data.close[0])
        # Buy all the available cash
        # self.order_target_value(target=self.broker.get_cash())
        #size = int(self.broker.get_cash() / self.data)
        #self.order = self.buy(size=size)
        self.buy()

    def stop(self):
        # calculate the actual returns
        self.roi = (self.broker.get_value() / self.val_start) - 1.0
        print('ROI:        {:.2f}%'.format(100.0 * self.roi))


if __name__ == '__main__':
    # Create a cerebro entitysetsizing
    cerebro = bt.Cerebro()

    datapath = os.path.join('../data/', 'SPX.csv')

    # Create a Data Feed
    data = bt.feeds.YahooFinanceCSVData(
        dataname=datapath,
        # Do not pass values before this date
        fromdate=datetime(2010, 1, 1),
        # Do not pass values before this date
        todate=datetime(2019, 12, 31),
        # Do not pass values after this date
        reverse=False)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    # Add a FixedSize sizer according to the stake
    # cerebro.addsizer(bt.sizers.FixedSize, stake=10)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.001)
    # cheat-on-close
    cerebro.broker.set_coc(True)          # doesn't seems to be working

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Add Analyzer
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

    # Add a strategy
    cerebro.addstrategy(BuyAndHold)

    # Run over everything
    results = cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.plot(style='candlestick')

    strat = results[0]
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
    import matplotlib.pyplot as plt
    perf_stats_strat = pf.timeseries.perf_stats(returns)
    drawdown_table = pf.timeseries.gen_drawdown_table(returns, 5)
    monthly_ret_table = ep.aggregate_returns(returns, 'monthly')
    monthly_ret_table = monthly_ret_table.unstack().round(3)
    ann_ret_df = pd.DataFrame(ep.aggregate_returns(returns, 'yearly'))
    ann_ret_df = ann_ret_df.unstack().round(3)
    print('-------------- PERFORMANCE ----------------')
    print(perf_stats_strat)
    print('-------------- DRAWDOWN ----------------')
    print(drawdown_table)
    print('-------------- MONTHLY RETURN ----------------')
    print(monthly_ret_table)
    print('-------------- ANNUAL RETURN ----------------')
    print(ann_ret_df)

    pf.create_full_tear_sheet(
        returns,
        positions=positions,
        transactions=transactions,
        #live_start_date='2005-05-01',
        round_trips=False)
    plt.show()