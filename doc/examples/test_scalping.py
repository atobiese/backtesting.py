import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
# from fin import fetch_download_data
from datetime import datetime
from backtesting import Strategy
from backtesting import Backtest
from backtesting.test import GOOG
from backtesting.lib import crossover
import logging
import os
from utilities import fetch_download_data
_LOG = logging.getLogger(__name__)
# run this to start logging
logging.debug("Begin")
_LOG.setLevel(logging.DEBUG)

dummy_data = True
if not dummy_data:
    # stck = '^GSPC'
    # stck = '^RUI'
    # stck = 'EURUSD=X'
    stck = 'AAPL'
    now = datetime.now()
    now_minus_10 = now + relativedelta(years=-10)
    # # get ticker data for the last 10  years
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Project root
    CSV_DIR = os.path.join(ROOT_DIR, "temp_csv")
    df = fetch_download_data(now_minus_10, now, stck, load_from_csv=True, csv_directory= CSV_DIR)
else:
    # use dummy data
    # stck = '^GSPC'
    # # stck = '^RUI'
    # # stck = 'EURUSD=X'
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Project root

    df = pd.read_csv(os.path.join(ROOT_DIR, "EURUSD_Candlestick_5_M_ASK_30.09.2019-30.09.2022.csv"))
    df = df[0:500]
    df["Gmt time"] = df["Gmt time"].str.replace(".000", "")
    df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S')
    df.set_index("Gmt time", inplace=True)
    df = df[df.High != df.Low]

# indicators
df['EMA200'] = ta.sma(df.Close, length=200)#sma ema
df['EMA20'] = ta.sma(df.Close, length=20)#sma ema
df["VWAP"] = ta.vwap(df.High, df.Low, df.Close, df.Volume)
df['RSI'] = ta.rsi(df.Close, length=16)
my_bbands = ta.bbands(df.Close, length=14, std=2.0)
df = df.join(my_bbands)
df['ATR'] = ta.atr(df.High, df.Low, df.Close, length=7)

# eventbased signals
def _vwap_sig_next(backcandles, dfa):
    sig = 0
    upt = 1
    dnt = 1
    for i in range(0, backcandles + 1):
        if max(dfa.Open[i], dfa.Close[i]) >= dfa.VWAP[i]:
            dnt = 0
        if min(dfa.Open[i], dfa.Close[i]) <= dfa.VWAP[i]:
            upt = 0
    if upt == 1 and dnt == 1:
        sig = 3
    elif upt == 1:
        sig = 2
    elif dnt == 1:
        sig = 1

    return sig

def _total_signal_next(l):
    if (df.vwap_sig[l] == 2
            and df.Close[l] <= df['BBL_14_2.0'][l]
            and df.RSI[l] < 45):
        return 2
    if (df.vwap_sig[l] == 1
            and df.Close[l] >= df['BBU_14_2.0'][l]
            and df.RSI[l] > 55):
        return 1
    return 0

# exactly_equal = pd.DataFrame.equals(df['vwap_sig'], df['vwap_sig_new'])
# assert(exactly_equal)

backcandles = 15

# vwap_sig = [0] * len(df)
vwap_sig = [0] * (backcandles)
nr_samples = len(df)

for row in range(backcandles, nr_samples):
    sig = _vwap_sig_next(backcandles, df[row - backcandles:row + 1])
    vwap_sig.append(sig)

df['vwap_sig'] = vwap_sig

total_sig = [0] * nr_samples
for row in range(backcandles, nr_samples):  # careful backcandles used previous cell
    total_sig[row] = _total_signal_next(row)
df['total_signal'] = total_sig

df[df.total_signal != 0].count()

def pointposbreak(x):
    if x['total_signal'] == 1:
        return x['High'] + 1e-4
    elif x['total_signal'] == 2:
        return x['Low'] - 1e-4
    else:
        return None

df['pointposbreak'] = df.apply(lambda row: pointposbreak(row), axis=1)

# dfpl = df[:75000].copy()
dfpl = df.copy()
# dfpl = df[1000:1500].copy()


indicators = ['VWAP', 'BBL_14_2.0', 'BBU_14_2.0', 'BBM_14_2.0']
colors = ['MediumPurple', 'orange',  'orange', 'black']
sub_indicators = ['RSI']
sub_colors = ['black']
from utilities import plot_visualization, update_plots_with_signals

metric_figure, fig, bottom_figure = plot_visualization(dfpl, indicators, colors, sub_indicators, sub_colors)
# get precalculated data
indicators = ['VWAP', 'RSI', 'BBL_14_2.0', 'BBU_14_2.0',
              'ATR', 'total_signal', 'pointposbreak', 'BBM_14_2.0', 'EMA200', 'EMA200']
colors = [None, None, 'red', 'red', None, None, 'green', 'blue', 'green', 'green']
overlay = [True, False, True, True, False, False, True, True, True, True]
scatter = [False, False, False, False, False, False, True, False, False, False]


from utilities import check_duration_on_exchange, get_data

class MyStrat(Strategy):
    initsize = 0.99
    mysize = initsize

    def init(self):
        super().init()

        for idx, p in enumerate(indicators):
            setattr(self, p, self.I(get_data, self.data, p, color=colors[idx], scatter=scatter[idx], overlay=overlay[idx]))

    def next(self):
        super().next()

        slatr = 1.2 * self.ATR[-1]
        TPSLRatio = 1.5
        today_idx = self.data._Data__i
        price_close = self.data.Close[-1]

        def is_long_close_signal():
            return sig is True if self.RSI[-1] >= 90 else False

        def is_short_close_signal():
            return sig is True if self.RSI[-1] <= 10 else False

        def is_long_buy_signal():
            return True if self.total_signal[-1] == 2 else False

        def is_short_sell_signal():
            return True if self.total_signal[-1] == 1 else False

        days_max_order_to_fulfill = 10
        check_duration_on_exchange(self, days_max_order_to_fulfill, today_idx)

        # determine if close trades
        if len(self.trades) > 0:
            if self.trades[-1].is_long and is_long_close_signal():
                self.trades[-1].close()
            elif self.trades[-1].is_short and is_short_close_signal():
                self.trades[-1].close()

        if len(self.trades) == 0 and is_long_buy_signal():
            sl1 = price_close - slatr
            tp1 = price_close + slatr * TPSLRatio
            self.buy(sl=sl1, tp=tp1, size=self.mysize, tag=today_idx)

        elif len(self.trades) == 0 and is_short_sell_signal():
            sl1 = price_close + slatr
            tp1 = price_close - slatr * TPSLRatio
            self.sell(sl=sl1, tp=tp1, size=self.mysize,tag=today_idx)


bt = Backtest(dfpl, MyStrat, cash=100, margin=1 / 10, commission=0.00)
stat = bt.run()
print(stat)
print(stat._trades)
# bt.plot(plot_width=1500)
bt.plot(plot_width=1500,smooth_equity=True, superimpose=True,reverse_indicators=False)
update_plots_with_signals(stat, metric_figure, fig, bottom_figure)
