# Backtest

Test against daily bars between 2010 and 2019. It may not be fair to some intraday strategies.

[Backtest Blog](https://letianzj.github.io/quanttrading-backtest.html)

[Examples of parameter search](./ma_double_cross.py).

[Examples of talib](./bollinger_bands.py)


|Index |Backtest                                                                         |Sharpe        |
|----:|:---------------------------------------------------------------------------------|-----------:|
|1 |  [Buy and Hold](./buy_hold.py)    | 0.78 |
|2 |  [Moving Average cross](./ma_cross.py)    | -0.27|
|3 |  [Moving Average double cross](./ma_double_cross.py)    | 0.32|
|4 |  [Bollinger Bands](./bollinger_bands.py)    |0.60 |
|5 |  [Dual Thrust](./dual_thrust.py)    | -0.10|
|6 |  [Ghost Trader](./ghost_trader.py)    | -0.14|
|7 |  [R-Breaker](./r_breaker.py)    | 0.15|
|8 |  [dynamic breakout ii](./dynamic_breakout_ii.py)    | 0.51|
|9 |  [Turtle](./turtle.py)    |0.15 |
|10 |  [Mebane Faber TAA](./mebane_faber_taa.py)    | 0.50|
|11 |  [MinVar, MaxSharpe, MaxDiversified, RiskParity](./portfolio_optimization.py)    | 1.08, 0.66, 0.90, 0.81|
|12 |  [Comdty Roll](./comdty_roll.py)    | pre-roll 0.55 vs last day -0.50 due to -$37.63 oil price|

```python

```