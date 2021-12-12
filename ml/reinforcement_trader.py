import numpy as np
import pandas as pd
import ta

from tensortrade.env.default import actions, rewards
from tensortrade.env.generic.components.informer import Informer
from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, Instrument
from tensortrade.oms.wallets import Wallet, Portfolio
import tensortrade.env.default as default
from tensortrade.env.generic import TradingEnv
from tensortrade.agents import DQNAgent


#-------------------- Data --------------------------#
symbol = 'SPY'
start_date = '2010-01-01'
end_date = '2021-06-30'

df = pd.read_csv(f'../data/{symbol}.csv', header=0, parse_dates=True, sep=',', index_col=0)
df.sort_index(ascending=True, inplace=True)
df = df[start_date:end_date]
df['Open'] = df['Adj Close'] / df['Close'] * df['Open']
df['High'] = df['Adj Close'] / df['Close'] * df['High']
df['Low'] = df['Adj Close'] / df['Close'] * df['Low']
df['Volume'] = df['Adj Close'] / df['Close'] * df['Volume']
df['Close'] = df['Adj Close']
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
price_history = df.copy()
price_history['date'] = df.index
ta.add_all_ta_features(df, 'Open', 'High', 'Low', 'Close', 'Volume', fillna=True)
df.columns = [c.lower() for c in df.columns]
price_history.columns = [c.lower() for c in price_history.columns]

#-------------------- Environment --------------------------#
# 1. exchange
exchange_name = "backtest-exchange"
exchange_options = ExchangeOptions(commission = 0.003, min_trade_size=1e-6)
exchange = Exchange(exchange_name, service=execute_order, options=exchange_options)(
    Stream.source(df['close'].tolist(), dtype="float").rename(f"USD-{symbol}")
)

# 2. data feed
with NameSpace(exchange_name):
    streams = [Stream.source(df[c].tolist(), dtype="float").rename(c) for c in df.columns]
feed = DataFeed(streams)
feed.next()

rstreams = [Stream.source(price_history[c].tolist()).rename(c) for c in price_history.columns]
rfeed = DataFeed(rstreams)

# 3. Portfolio
SPY = Instrument(symbol, 1, symbol)
portfolio = Portfolio(USD, [
    Wallet(exchange, 1_000_000 * USD),
    Wallet(exchange, 0 * SPY),
])

# 4. TradingEnv
# TODO net_worth in external data source
chart_renderer = default.renderers.PlotlyTradingChart()
screen_logger = default.renderers.ScreenLogger(date_format="%Y-%m-%d %H:%M:%S %p")

env = default.create(
    portfolio=portfolio,
    action_scheme=default.actions.SimpleOrders(trade_sizes=[1.0], min_order_pct=0),       # 100% buy or sell
    reward_scheme=default.rewards.SimpleProfit(window_size=1),
    feed=feed,
    renderer_feed=rfeed,
    window_size=22,
        renderer=[
      chart_renderer,
      screen_logger,
    ]
)

#-------------------- Training --------------------------#
# 1. random policy
done = False
obs = env.reset()
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
#print(portfolio.ledger.as_frame().head(20))
#portfolio.ledger.as_frame().to_clipboard(index=False)
#env.render()

# 2. DQN agent
obs = env.reset()
agent = DQNAgent(env)
agent.train(n_episodes=2, n_steps=200, render_interval=10)

performance = pd.DataFrame.from_dict(portfolio.performance, orient='index')
performance.plot()
performance.net_worth.plot()