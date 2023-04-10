from backtesting import Strategy
from backtesting import Backtest
from backtesting.test import GOOG
from backtesting.lib import crossover, cross

class DCA(Strategy):
    buy_dollars = 1000
    buy_period = 100
    count = 0
    cash_total = 0

    def init(self):
        self.cash_total = self.equity

    def next(self):
        self.count += 1
        if not (self.count % self.buy_period):
            cash_left = self.cash_total - sum(t.entry_price * t.size for t in self.trades)
            if (self.buy_dollars >= self.data.Close[-1]):
                self.buy(size=self.buy_dollars // self.data.Close[-1])


bt = Backtest(GOOG, DCA, cash=100_000, commission=0)
stat= bt.run()
print(stat)

print(stat._trades)
print(stat._strategy)

bt.plot(plot_width=1500, smooth_equity=False,
        superimpose=False, reverse_indicators=False,
        open_browser=True, plot_drawdown=False)