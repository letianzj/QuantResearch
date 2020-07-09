'''
	It is based off of Mebane Faber's "Global Tactical Asset Allocation" (GTAA). GTAA consists of five global asset
    classes:  US stocks, foreign stocks, bonds, real estate and commodities...it is either long the asset class or in cash
    with its allocation of the funds.

    The basics of the strategy go like this:
    https://www.quantopian.com/posts/meb-fabers-global-tactical-asset-allocation-gtaa-strategy
    (1) Look at a 200 day or 10 months trailing window (SA - Slow Average) versus a 20 day trailing window (FA - Fast Average).
    (2) If the FA is greater than the SA, go long about 20% of your portfolio in that security
    (3) If the FA is less than the SA, have 0% of your portfolio in that security; hold cash instead
    The system updates monthly at the end of the month.

    In this historical bull markets, nothing beats simply holding SPY. But the strategy didn't beat naive diversification 20% holdings.
    It avoid downturn end of 2015 by holding all cash; yet missed wave 3 that began at 2016.
'''
import os
import numpy as np
import pandas as pd
from datetime import datetime
import backtrader as bt
from IPython.core.display import display, HTML
# set browser full width
display(HTML("<style>.container { width:100% !important; }</style>"))


class EndOfMonth(object):
    def __init__(self, cal):
        self.cal = cal

    def __call__(self, d):
        if self.cal.last_monthday(d):
            return True
        return False


class MebaneFaberTAA(bt.Strategy):
    params = (
        ('nslow', 200),
        ('nfast', 20),
        ('printlog', False),        # comma is required
    )

    def __init__(self):
        self.buyprice = None
        self.buycomm = None
        self.bar_executed = None
        self.val_start = None

        self.add_timer(
            when=bt.Timer.SESSION_START,       # before next
            allow=EndOfMonth(cal = bt.TradingCalendar())
        )

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

    def next(self):
        pass

    def notify_timer(self, timer, when, *args, **kwargs):
        print('{} strategy notify_timer with tid {}, when {} cheat {}'.
              format(self.data.datetime.datetime(), timer.p.tid, when, timer.p.cheat))
        if len(self.datas[0]) < self.p.nslow:       # not enough bars
            return

        total_value = self.broker.getvalue()
        stock_value = total_value * 0.95 / 5
        for data in self.datas:
            pos = self.getposition(data).size
            ma_fast = np.mean(data.get(0, self.p.nfast))
            ma_slow = np.mean(data.get(0, self.p.nslow))
            if ma_fast > ma_slow:         # buy
                target_pos = (int)(stock_value / data.close[0])
                self.order_target_size(data=data, target=target_pos)
                self.log('LONG ORDER SENT, %s, Price: %.2f, fast sma: %.2f, slow sma: %.2f, Size: %.2f' %
                         (data._name,
                          data.close[0],
                          ma_fast,
                          ma_slow,
                          target_pos))
            else:       # hold cash
                target_pos = 0
                self.order_target_size(data=data, target=target_pos)
                self.log('FLAT ORDER SENT, %s, Price: %.2f, fast sma: %.2f, slow sma: %.2f, Size: %.2f' %
                         (data._name,
                          data.close[0],
                          ma_fast,
                          ma_slow,
                          target_pos))

    def stop(self):
        # calculate the actual returns
        print(self.analyzers)
        roi = (self.broker.get_value() / self.val_start) - 1.0
        self.log('ROI:        {:.2f}%'.format(100.0 * roi))
        self.log('(Mebane Faber TAA Ending Value %.2f' %
                  self.broker.getvalue(), doprint=True)


if __name__ == '__main__':
    param_opt = False
    perf_eval = True
    initial_capital = 100000.0
    etfs = ['SPY', 'EFA', 'TIP', 'GSG', 'VNQ']
    benchmark = etfs

    cerebro = bt.Cerebro()

    # Add the Data Feed to Cerebro
    # SPY: S&P 500
    # EFA: MSCI EAFE
    # TIP: UST
    # GSG: GSCI
    # VNQ: REITs
    for s in etfs:
        # Create a Data Feed
        data = bt.feeds.YahooFinanceCSVData(
            dataname=os.path.join('../data/', f'{s}.csv'),
            fromdate=datetime(2010, 1, 1),
            todate=datetime(2019, 12, 31),
            reverse=False)
        cerebro.adddata(data, name=s)

    # Set our desired cash start
    cerebro.broker.setcash(initial_capital)

    # Add a FixedSize sizer according to the stake
    # cerebro.addsizer(bt.sizers.FixedSize, stake=10)
    # PercentSizer will flat position first; overwrite if not desired.
    # cerebro.addsizer(bt.sizers.PercentSizerInt, percents=95)

    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.001)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Add a strategy
    cerebro.addstrategy(MebaneFaberTAA, printlog=True)

    # Add Analyzer
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='SharpeRatio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='DrawDown')
    cerebro.addanalyzer(bt.analyzers.PositionsValue, _name='positions', cash=True)
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
        # somehow pyfolio analyzer doesn't handle well multi-assets
        df_positions = pd.DataFrame.from_dict(strat.analyzers.positions.get_analysis(), orient='index')
        df_positions.columns = etfs+['cash']

        print('-------------- RETURNS ----------------')
        print(returns)
        print('-------------- POSITIONS ----------------')
        print(df_positions)
        print('-------------- TRANSACTIONS ----------------')
        print(transactions)
        print('-------------- GROSS LEVERAGE ----------------')
        print(gross_lev)

        import empyrical as ep
        import pyfolio as pf

        bm_ret = None
        returns = returns[transactions.index[0]:]  # count from first trade
        # returns.index = returns.index.tz_localize(None)  # # remove tzinfo; tz native
        df_positions.index = df_positions.index.map(lambda x: datetime.combine(x, datetime.min.time()))
        df_positions.index = df_positions.index.tz_localize('UTC')
        df_positions = df_positions.loc[returns.index]

        df_constituents = pd.DataFrame()
        for s in ['SPY', 'EFA', 'TIP', 'GSG', 'VNQ']:
            datapath = os.path.join('../data/', f'{s}.csv')
            df_temp = pd.read_csv(datapath, index_col=0)
            df_temp = df_temp['Adj Close']
            df_temp.name = s
            df_constituents = pd.concat([df_constituents, df_temp], axis=1)

        df_constituents_ret = df_constituents.pct_change()
        df_constituents_ret.index = pd.to_datetime(df_constituents_ret.index)
        df_constituents_ret.index = df_constituents_ret.index.tz_localize('UTC')
        df_constituents_ret = df_constituents_ret.loc[returns.index]
        df_constituents_ret['Benchmark'] = df_constituents_ret.mean(axis=1)  # 20% each
        df_constituents_value = initial_capital * (df_constituents_ret + 1).cumprod()

        perf_stats_strat = pf.timeseries.perf_stats(returns)
        perf_stats_bm = pf.timeseries.perf_stats(df_constituents_ret.Benchmark)
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
            benchmark_rets=df_constituents_ret['Benchmark'],
            positions=df_positions,
            transactions=transactions,
            #live_start_date='2005-05-01',
            round_trips=False)
        plt.show()

        df_value_all = df_constituents_value.copy()
        # df_value_all =df_constituents_value.merge(df_positions['Total'], how='inner', left_index=True, right_index=True)
        df_value_all['Strategy'] = df_positions.sum(axis=1)
        df_value_all.plot()
        plt.show()