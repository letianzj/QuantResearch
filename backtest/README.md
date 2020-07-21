# Backtest

Test against daily bars between 2010 and 2019, using [Backtrader](https://www.backtrader.com/). It may not be fair to some intraday strategies.

|Index |Backtest                                                                         |Sharpe        |
|----:|:---------------------------------------------------------------------------------|-----------:|
|1 |  [Buy and Hold](./buy_hold.py)    | 0.78 |
|2 |  [Moving Average cross](./ma_cross.py)    | -0.51|
|3 |  [Moving Average double cross](./ma_double_cross.py)    | 0.39|
|4 |  [Bollinger Bands](./bollinger_bands.py)    |0.46 |
|5 |  [Dual Thrust](./dual_thrust.py)    | -0.55|
|6 |  [Ghost Trader](./ghost_trader.py)    | -0.74|
|7 |  [R-Breaker](./r_breaker.py)    | -0.22|
|8 |  [Mebane Faber TAA](./mebane_faber_taa.py)    | 0.40|
|9 |  [MinVar, MaxSharpe, MaxDiversified, RiskParity](./portfolio_optimization.py)    | 0.93, 0.37, 0.71, 0.66|

```python

```