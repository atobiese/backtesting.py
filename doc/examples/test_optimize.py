from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import numpy as np
from backtesting.test import SMA, GOOG


class SmaCross(Strategy):
    # NOTE: These values are also used on the website!
    fast = 10
    slow = 30

    def init(self):
        import pandas_ta as ta
        self.sma1 = self.I(ta.sma, self.data.Close.df.Close, self.fast)
        self.sma2 = self.I(ta.sma, self.data.Close.df.Close, self.slow)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell()


bt = Backtest(GOOG, SmaCross, commission=.002,
              exclusive_orders=True)
stats = bt.run()
bt.plot(plot_width=1500, plot_drawdown=True, open_browser=True)
#
OPT_PARAMS = {'fast': range(2, 5, 2), 'slow': [2, 5, 7, 9]}
res = bt.optimize(**OPT_PARAMS)
# res3, heatmap = bt.optimize(**OPT_PARAMS, return_heatmap=True,
#                                     constraint=lambda d: d.slow > 2 * d.fast)
print(stats)
print(res)
# print(res)
# print(res3)

# bt = Backtest(GOOG.iloc[:100], SmaCross)
# res, heatmap, skopt_results = bt.optimize(
#     fast=range(2, 20), slow=np.arange(2, 20, dtype=object),
#     constraint=lambda p: p.fast < p.slow,
#     max_tries=30,
#     method='skopt',
#     return_optimization=True,
#     return_heatmap=True,
#     random_state=2)
# a=1
# import seaborn as sns
#
#
# sns.heatmap(heatmap[::-1], cmap='viridis')

