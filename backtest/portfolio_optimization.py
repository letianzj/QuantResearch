'''
This is a follow up of https://letianzj.github.io/portfolio-management-one.html
It backtests four portfolios: GMV, tangent, maximum diversification and risk parity
and compare them with equally-weighted portfolio
'''
import os
import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timezone
import quanttrader as qt
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import empyrical as ep
import pyfolio as pf
# set browser full width
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

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


class PortfolioOptimization(qt.StrategyBase):
    def __init__(self, nlookback=200, model='gmv'):
        super(PortfolioOptimization, self).__init__()
        self.nlookback = nlookback,
        self.model = model
        self.current_time = None

    def on_tick(self, tick_event):
        self.current_time = tick_event.timestamp
        # print('Processing {}'.format(self.current_time))

        # wait for enough bars
        for symbol in self.symbols:
            df_hist = self._data_board.get_hist_price(symbol, self.current_time)
            if df_hist.shape[0] < self.nlookback:
                return

        # wait for month end
        time_index = self._data_board.get_hist_time_index()
        time_loc = time_index.get_loc(self.current_time)
        if (time_loc != len(time_index)-1) & (time_index[time_loc].month == time_index[time_loc+1].month):
            return

        npv = self._position_manager.current_total_capital
        n_stocks = len(self.symbols)
        TOL = 1e-12
        prices = None
        for symbol in self.symbols:
            price = self._data_board.get_hist_price(symbol, self.current_time)['Close'].iloc[-self.nlookback:]
            price = np.array(price)
            if prices is None:
                prices = price
            else:
                prices = np.c_[prices, price]
        rets = prices[1:,:]/prices[0:-1, :]-1.0
        mu = np.mean(rets, axis=0)
        cov = np.cov(rets.T)

        w = np.ones(n_stocks) / n_stocks  # default
        try:
            if self.model == 'gmv':
                w0 = np.ones(n_stocks) / n_stocks
                cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}, {'type': 'ineq', 'fun': lambda w: w})
                res = minimize(minimum_vol_obj, w0, args=cov, method='SLSQP', constraints=cons, tol=TOL, options={'disp': True})
                if not res.success:
                    print(f'{self.model} Optimization failed')
                w = res.x
            elif self.model == 'sharpe':
                w0 = np.ones(n_stocks) / n_stocks
                cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}, {'type': 'ineq', 'fun': lambda w: w})
                res = minimize(maximum_sharpe_negative_obj, w0, args=[mu, cov], method='SLSQP', constraints=cons, tol=TOL, options={'disp': True})
                w = res.x
            elif self.model == 'diversified':
                w0 = np.ones(n_stocks) / n_stocks
                cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})  # weights sum to one
                bnds = tuple([(0, 1)] * n_stocks)
                res = minimize(maximum_diversification_negative_obj, w0, bounds=bnds, args=cov, method='SLSQP', constraints=cons, tol=TOL, options={'disp': True})
                w = res.x
            elif self.model == 'risk_parity':
                w0 = np.ones(n_stocks) / n_stocks
                w_b = np.ones(n_stocks) / n_stocks  # risk budget/target, percent of total portfolio risk (in this case equal risk)
                # bnds = ((0,1),(0,1),(0,1),(0,1)) # alternative, use bounds for weights, one for each stock
                cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}, {'type': 'ineq', 'fun': lambda x: x})
                res = minimize(risk_budget_obj, w0, args=[cov, w_b], method='SLSQP', constraints=cons, tol=TOL, options={'disp': True})
                w = res.x
        except Exception as e:
            print(f'{self.model} Optimization failed; {str(e)}')

        i = 0
        for sym in self.symbols:
            current_size = self._position_manager.get_position_size(sym)
            current_price = self._data_board.get_hist_price(sym, self.current_time)['Close'].iloc[-1]
            target_size = (int)(npv * w[i] / current_price)
            self.adjust_position(sym, size_from=current_size, size_to=target_size, timestamp=self.current_time)
            print('REBALANCE ORDER SENT, %s, Price: %.2f, Percentage: %.2f, Target Size: %.2f' %
                         (sym,
                          current_price,
                          w[i],
                          target_size))
            i += 1


if __name__ == '__main__':
    etfs = ['SPY', 'EFA', 'TIP', 'GSG', 'VNQ']
    models = ['gmv', 'sharpe', 'diversified', 'risk_parity']
    benchmark = etfs
    init_capital = 100_000.0
    test_start_date = datetime(2010,1,1, 8, 30, 0, 0, pytz.timezone('America/New_York'))
    test_end_date = datetime(2019,12,31, 6, 0, 0, 0, pytz.timezone('America/New_York'))

    dict_results = dict()
    for model in models:
        dict_results[model] = dict()

        # SPY: S&P 500
        # EFA: MSCI EAFE
        # TIP: UST
        # GSG: GSCI
        # VNQ: REITs
        strategy = PortfolioOptimization()
        strategy.set_capital(init_capital)
        strategy.set_symbols(etfs)
        strategy.set_params({'nlookback': 200, 'model': model})

        backtest_engine = qt.BacktestEngine(test_start_date, test_end_date)
        backtest_engine.set_capital(init_capital)  # capital or portfolio >= capital for one strategy

        for symbol in etfs:
            data = qt.util.read_ohlcv_csv(os.path.join('../data/', f'{symbol}.csv'))
            backtest_engine.add_data(symbol, data)

        backtest_engine.set_strategy(strategy)

        ds_equity, df_positions, df_trades = backtest_engine.run()
        # save to excel
        qt.util.save_one_run_results('./output', ds_equity, df_positions, df_trades, batch_tag=model)

        ds_ret = ds_equity.pct_change().dropna()
        ds_ret.name = model
        dict_results[model]['equity'] = ds_equity
        dict_results[model]['return'] = ds_ret
        dict_results[model]['positions'] = df_positions
        dict_results[model]['transactions'] = df_trades

    # ------------------------- Evaluation and Plotting -------------------------------------- #
    bm = pd.DataFrame()
    for s in etfs:
        df_temp = qt.util.read_ohlcv_csv(os.path.join('../data/', f'{s}.csv'))
        df_temp = df_temp['Close']
        df_temp.name = s
        bm = pd.concat([bm, df_temp], axis=1)

    bm_ret = bm.pct_change().dropna()
    bm_ret.index = pd.to_datetime(bm_ret.index)
    bm_ret = bm_ret.loc[dict_results[models[0]]['return'].index]
    bm_ret['benchmark'] = bm_ret.mean(axis=1)  # 20% each
    bm_value = init_capital * (bm_ret + 1).cumprod()

    perf_stats_all = pd.DataFrame()
    for m in models:
        perf_stats_strat = pf.timeseries.perf_stats(dict_results[m]['return'])
        perf_stats_strat.name = m
        perf_stats_all = pd.concat([perf_stats_all, perf_stats_strat], axis=1)
    perf_stats_bm = pf.timeseries.perf_stats(bm_ret.benchmark)
    perf_stats_bm.name = 'equal_weights'
    perf_stats_all = pd.concat([perf_stats_all, perf_stats_bm], axis=1)
    print(perf_stats_all)

    # portfolio values
    portfolio_value_all = pd.DataFrame()
    for m in models:
        port_value = dict_results[m]['positions'].sum(axis=1)
        port_value.name = m
        portfolio_value_all = pd.concat([portfolio_value_all, port_value], axis=1)
    port_value = bm_value.benchmark.copy()
    port_value.name = 'equal_weights'
    portfolio_value_all = pd.concat([portfolio_value_all, port_value], axis=1)
    fig, ax = plt.subplots(2, 1, figsize=(5, 12))
    portfolio_value_all.plot(ax=ax[0])
    bm_value[etfs].plot(ax=ax[1])
    fig.tight_layout()
    plt.show()

    # monthly returns
    fig, ax = plt.subplots(5, 1, figsize=(10, 35))
    i = 0
    for m in models:
        pf.plotting.plot_monthly_returns_heatmap(dict_results[m]['return'], ax[i])
        ax[i].title.set_text(m)
        i += 1
    pf.plotting.plot_monthly_returns_heatmap(bm_ret['benchmark'], ax[i])
    ax[i].title.set_text('equal weighted')
    fig.tight_layout()
    plt.show()

    # positions
    fig, ax = plt.subplots(4, 1, figsize=(25, 25))
    etfs_plus_cash = etfs+['cash']
    i = 0
    for m in models:
        sum_ = dict_results[m]['positions'].sum(axis=1)
        pcts = []
        for etf in etfs_plus_cash:
            pct = dict_results[m]['positions'][etf] / sum_
            pcts.append(pct)
        print(pcts[0].shape, len(pcts))
        ax[i].stackplot(pcts[0].index, pcts, labels=etfs_plus_cash)
        ax[i].legend(loc='upper left')
        ax[i].title.set_text(m)
        i += 1
    fig.tight_layout()
    plt.show()