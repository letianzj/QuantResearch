'''
This is a follow up of https://letianzj.github.io/portfolio-management-one.html
It backtests four portfolios: GMV, tangent, maximum diversification and risk parity
and compare them with equally-weighted portfolio
'''
import os
import numpy as np
import pandas as pd
from datetime import datetime
import backtrader as bt
from scipy.optimize import minimize
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

# ------------------ help functions -------------------------------- #
def minimum_vol_obj(wo, cov):
    w = wo.reshape(-1, 1)
    sig_p = np.sqrt(np.matmul(w.T, np.matmul(cov, w)))[0, 0]    # portfolio sigma
    return sig_p

def maximum_sharpe_negative_obj(wo, mu_cov):
    w = wo.reshape(-1, 1)
    mu = mu_cov[0].reshape(-1, 1)
    cov = mu_cov[1]
    obj = np.matmul(w.T, mu)[0, 0]
    sig_p = np.sqrt(np.matmul(w.T, np.matmul(cov, w)))[0, 0]    # portfolio sigma
    obj = -1 * obj/sig_p
    return obj

def maximum_diversification_negative_obj(wo, cov):
    w = wo.reshape(-1, 1)
    w_vol = np.matmul(w.T, np.sqrt(np.diag(cov).reshape(-1, 1)))[0, 0]
    port_vol = np.sqrt(np.matmul(w.T, np.matmul(cov, w)))[0, 0]
    ratio = w_vol / port_vol
    return -ratio

# this is also used to verify rc from optimal w
def calc_risk_contribution(wo, cov):
    w = wo.reshape(-1, 1)
    sigma = np.sqrt(np.matmul(w.T, np.matmul(cov, w)))[0, 0]
    mrc = np.matmul(cov, w)
    rc = (w * mrc) / sigma  # element-wise multiplication
    return rc

def risk_budget_obj(wo, cov_wb):
    w = wo.reshape(-1, 1)
    cov = cov_wb[0]
    wb = cov_wb[1].reshape(-1, 1)  # target/budget in percent of portfolio risk
    sig_p = np.sqrt(np.matmul(w.T, np.matmul(cov, w)))[0, 0]  # portfolio sigma
    risk_target = sig_p * wb
    asset_rc = calc_risk_contribution(w, cov)
    f = np.sum(np.square(asset_rc - risk_target.T))  # sum of squared error
    return f

class PortfolioOptimization(bt.Strategy):
    params = (
        ('nlookback', 200),
        ('model', 'gmv'),         # gmv, sharpe, diversified, risk_parity
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
        print(f'================================== start portfolio {self.p.model} ======================================')

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
        if len(self.datas[0]) < self.p.nlookback:       # not enough bars
            return

        total_value = self.broker.getvalue()
        i = 0
        prices = None
        for data in self.datas:
            price = data.close.get(0, self.p.nlookback)
            price = np.array(price)
            if i == 0:
                prices = price
            else:
                prices = np.c_[prices, price]
            i += 1
        rets = prices[1:,:]/prices[0:-1, :]-1.0
        mu = np.mean(rets, axis=0)
        cov = np.cov(rets.T)

        n_stocks = len(self.datas)
        TOL = 1e-12
        w = np.ones(n_stocks) / n_stocks      # default
        try:
            if self.p.model == 'gmv':
                w0 = np.ones(n_stocks) / n_stocks
                cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}, {'type': 'ineq', 'fun': lambda w: w})
                res = minimize(minimum_vol_obj, w0, args=cov, method='SLSQP', constraints=cons, tol=TOL, options={'disp': True})
                if not res.success:
                    self.log(f'{self.p.model} Optimization failed')
                w = res.x
            elif self.p.model == 'sharpe':
                w0 = np.ones(n_stocks) / n_stocks
                cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}, {'type': 'ineq', 'fun': lambda w: w})
                res = minimize(maximum_sharpe_negative_obj, w0, args=[mu, cov], method='SLSQP', constraints=cons, tol=TOL, options={'disp': True})
                w = res.x
            elif self.p.model == 'diversified':
                w0 = np.ones(n_stocks) / n_stocks
                cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})  # weights sum to one
                bnds = tuple([(0, 1)] * n_stocks)
                res = minimize(maximum_diversification_negative_obj, w0, bounds=bnds, args=cov, method='SLSQP', constraints=cons, tol=TOL, options={'disp': True})
                w = res.x
            elif self.p.model == 'risk_parity':
                w0 = np.ones(n_stocks) / n_stocks
                w_b = np.ones(n_stocks) / n_stocks  # risk budget/target, percent of total portfolio risk (in this case equal risk)
                # bnds = ((0,1),(0,1),(0,1),(0,1)) # alternative, use bounds for weights, one for each stock
                cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}, {'type': 'ineq', 'fun': lambda x: x})
                res = minimize(risk_budget_obj, w0, args=[cov, w_b], method='SLSQP', constraints=cons, tol=TOL, options={'disp': True})
                w = res.x
        except Exception as e:
            self.log(f'{self.p.model} Optimization failed; {str(e)}')

        stock_value = total_value * 0.95
        i = 0
        for data in self.datas:
            target_pos = (int)(stock_value * w[i] / data.close[0])
            self.order_target_size(data=data, target=target_pos)
            self.log('REBALANCE ORDER SENT, %s, Price: %.2f, Percentage: %.2f, Target Size: %.2f' %
                         (data._name,
                          data.close[0],
                          w[i],
                          target_pos))
            i += 1

    def stop(self):
        # calculate the actual returns
        print(self.analyzers)
        roi = (self.broker.get_value() / self.val_start) - 1.0
        self.log('ROI:        {:.2f}%'.format(100.0 * roi))
        self.log(f'{self.p.model} ending Value {self.broker.getvalue():.2f}', doprint=True)


if __name__ == '__main__':
    param_opt = False
    perf_eval = True
    initial_capital = 100000.0
    etfs = ['SPY', 'EFA', 'TIP', 'GSG', 'VNQ']
    benchmark = etfs

    strategies = ['gmv', 'sharpe', 'diversified', 'risk_parity']
    dict_results = dict()
    for sname in strategies:
        dict_results[sname] = dict()
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

        # Set the commission - 0.1% ... divide by 100 to remove the %
        cerebro.broker.setcommission(commission=0.001)

        # Print out the starting conditions
        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

        # Add a strategy
        cerebro.addstrategy(PortfolioOptimization, model=sname, printlog=True)

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

        pyfoliozer = strat.analyzers.getbyname('pyfolio')
        returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
        # somehow pyfolio analyzer doesn't handle well multi-assets
        df_positions = pd.DataFrame.from_dict(strat.analyzers.positions.get_analysis(), orient='index')
        df_positions.columns = etfs+['cash']
        returns = returns[transactions.index[0]:]  # count from first trade
        # returns.index = returns.index.tz_localize(None)  # # remove tzinfo; tz native
        df_positions.index = df_positions.index.map(lambda x: datetime.combine(x, datetime.min.time()))
        df_positions.index = df_positions.index.tz_localize('UTC')
        df_positions = df_positions.loc[returns.index]

        # save immediate results
        dict_results[sname]['returns'] = returns
        dict_results[sname]['positions'] = df_positions
        dict_results[sname]['transactions'] = transactions

    # Compare four portfolios with equal weighted
    import matplotlib.pyplot as plt
    import empyrical as ep
    import pyfolio as pf

    bm_ret = None
    df_constituents = pd.DataFrame()
    for s in etfs:
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

    # return stats
    perf_stats_all = pd.DataFrame()
    for s in strategies:
        perf_stats_strat = pf.timeseries.perf_stats(dict_results[s]['returns'])
        perf_stats_strat.name = s
        perf_stats_all = pd.concat([perf_stats_all, perf_stats_strat], axis=1)
    perf_stats_bm = pf.timeseries.perf_stats(df_constituents_ret.Benchmark)
    perf_stats_bm.name = 'equal_weights'
    perf_stats_all = pd.concat([perf_stats_all, perf_stats_bm], axis=1)
    print(perf_stats_all)

    # portfolio values
    portfolio_value_all = pd.DataFrame()
    for s in strategies:
        port_value = dict_results[s]['positions'].sum(axis=1)
        port_value.name = s
        portfolio_value_all = pd.concat([portfolio_value_all, port_value], axis=1)
    port_value = df_constituents_value.Benchmark.copy()
    port_value.name = 'equal_weights'
    portfolio_value_all = pd.concat([portfolio_value_all, port_value], axis=1)
    fig, ax = plt.subplots(2, 1, figsize=(5, 12))
    portfolio_value_all.plot(ax=ax[0])
    df_constituents_value[etfs].plot(ax=ax[1])
    fig.tight_layout()
    plt.show()

    # monthly returns
    fig, ax = plt.subplots(5, 1, figsize=(10, 35))
    i = 0
    for s in strategies:
        pf.plotting.plot_monthly_returns_heatmap(dict_results[s]['returns'], ax[i])
        ax[i].title.set_text(s)
        i += 1
    pf.plotting.plot_monthly_returns_heatmap(df_constituents_ret['Benchmark'], ax[i])
    ax[i].title.set_text('equal weighted')
    fig.tight_layout()
    plt.show()

    # positions
    fig, ax = plt.subplots(4, 1, figsize=(25, 25))
    i = 0
    for s in strategies:
        sum_ = dict_results[s]['positions'].sum(axis=1)
        pcts = []
        for etf in etfs:
            pct = dict_results[s]['positions'][etf] / sum_
            pcts.append(pct)
        ax[i].stackplot(dict_results[s]['positions'].index, pcts, labels=etfs)
        ax[i].legend(loc='upper left')
        ax[i].title.set_text(s)
        i += 1
    fig.tight_layout()
    plt.show()