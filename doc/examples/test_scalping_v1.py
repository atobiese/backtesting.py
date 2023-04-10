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
    df = fetch_download_data(now_minus_10, now, stck, load_from_csv=True, csv_directory=CSV_DIR)
else:
    # use dummy data
    # stck = '^GSPC'
    # # stck = '^RUI'
    # # stck = 'EURUSD=X'
    # ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Project root
    #
    # df = pd.read_csv(os.path.join(ROOT_DIR, "EURUSD_Candlestick_5_M_ASK_30.09.2019-30.09.2022.csv"))
    # df = df[0:500]
    # df["Gmt time"] = df["Gmt time"].str.replace(".000", "")
    # df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S')
    # df.set_index("Gmt time", inplace=True)
    # df = df[df.High != df.Low]
    df = yf.download("^RUI", start='2011-01-05', end='2021-01-05')
    df = df[df.High != df.Low]
    n = 0
    df.drop(index=df.index[:n], inplace=True)
    # df.reset_index(inplace=True)
    # print(df)

# indicators
df['EMA200'] = ta.ema(df.Close, length=200)  # sma ema
df['EMA20'] = ta.ema(df.Close, length=20)  # sma ema
# df["VWAP"] = ta.vwap(df.High, df.Low, df.Close, df.Volume)
df['RSI'] = ta.rsi(df.Close, length=12)
my_bbands = ta.bbands(df.Close, length=14, std=2.0)
df = df.join(my_bbands)
df['ATR'] = ta.atr(df.High, df.Low, df.Close, length=7)
# KST osciallator (momentum)
_df_KST = ta.kst(df.Close, 10, 15, 20, 30, 10, 10, 10, 15, 9)
df = df.join(_df_KST)


# strategy:
#
def addema200signal(df, backcandles):
    emasignal = [0] * len(df)
    for row in range(backcandles, len(df)):
        upt = 1
        dnt = 1
        for i in range(row - backcandles, row + 1):
            if df.High[i] >= df['EMA200'][i]:
                dnt = 0
            if df.Low[i] <= df['EMA200'][i]:
                upt = 0
        if upt == 1 and dnt == 1:
            # print("!!!!! check trend loop !!!!")
            emasignal[row] = 3
        elif upt == 1:
            emasignal[row] = 2
        elif dnt == 1:
            emasignal[row] = 1
    df['ema200_signal'] = emasignal


nr_for_confirmation = 6
addema200signal(df, nr_for_confirmation)


# eventbased signals
# def _vwap_sig_next(backcandles, dfa):
#     sig = 0
#     upt = 1
#     dnt = 1
#     for i in range(0, backcandles + 1):
#         if max(dfa.Open[i], dfa.Close[i]) >= dfa.VWAP[i]:
#             dnt = 0
#         if min(dfa.Open[i], dfa.Close[i]) <= dfa.VWAP[i]:
#             upt = 0
#     if upt == 1 and dnt == 1:
#         sig = 3
#     elif upt == 1:
#         sig = 2
#     elif dnt == 1:
#         sig = 1
#
#     return sig


# def _total_signal_next(l):
#     if (df.vwap_sig[l] == 2
#             and df.Close[l] <= df['BBL_14_2.0'][l]
#             and df.RSI[l] < 45):
#         return 2
#     if (df.vwap_sig[l] == 1
#             and df.Close[l] >= df['BBU_14_2.0'][l]
#             and df.RSI[l] > 55):
#         return 1
#     return 0


# exactly_equal = pd.DataFrame.equals(df['vwap_sig'], df['vwap_sig_new'])
# assert(exactly_equal)

# backcandles = 15

## vwap_sig = [0] * len(df)
# vwap_sig = [0] * (backcandles)
# nr_samples = len(df)
#
# for row in range(backcandles, nr_samples):
#     sig = _vwap_sig_next(backcandles, df[row - backcandles:row + 1])
#     vwap_sig.append(sig)
#
# df['vwap_sig'] = vwap_sig


# total_sig = [0] * nr_samples
# for row in range(backcandles, nr_samples):  # careful backcandles used previous cell
#     total_sig[row] = _total_signal_next(row)
# df['total_signal'] = total_sig


def addorderslimit(df, percent):
    ordersignal = [0] * len(df)
    for i in range(1, len(df)):  # EMASignal of previous candle!!! modified!!!
        if df.Close[i] <= df['BBL_14_2.0'][i] and  df['ema200_signal'][i] == 2:  # and df.RSI[i]<=100: #Added RSI condition to avoid direct close condition
            ordersignal[i] = df.Close[i] - df.Close[i] * percent
        elif df.Close[i] >= df['BBU_14_2.0'][i] and df['ema200_signal'][i] == 1:  # and df.RSI[i]>=0:
            ordersignal[i] = df.Close[i] + df.Close[i] * percent
    df['total_signal'] = ordersignal
    return df


df = addorderslimit(df, 0.00)


# df[df.total_signal != 0].count()

def pointposbreak(x):
    # if x['total_signal'] == 1:
    #     return x['High'] + 1e-4
    # elif x['total_signal'] == 2:
    #     return x['Low'] - 1e-4
    if x['total_signal']!=0:
        return x['total_signal']
    else:
        return None

df['pointposbreak'] = df.apply(lambda row: pointposbreak(row), axis=1)

# dfpl = df[:75000].copy()
dfpl = df.copy()
# dfpl = df[1000:1500].copy()


indicators = ['EMA200', 'BBL_14_2.0', 'BBU_14_2.0', 'BBM_14_2.0', 'total_signal']
colors = ['MediumPurple', 'orange', 'orange', 'black',  'black']
sub_indicators = ['RSI', 'ema200_signal']
sub_colors = ['black', 'green']

from utilities import plot_visualization, update_plots_with_signals

# dfpl = df[:75000].copy()
dfpl = df.copy()
dfpl_plot = df[1000:1200].copy()

metric_figure, fig, bottom_figure = plot_visualization(dfpl_plot, indicators, colors, sub_indicators, sub_colors)
# metric_figure.show()
# get precalculated data
indicators = ['RSI', 'BBL_14_2.0', 'BBU_14_2.0',
              'ATR', 'total_signal', 'pointposbreak', 'BBM_14_2.0', 'EMA200', 'EMA20', 'ema200_signal']
colors = [ None, 'red', 'red', None, None, 'green', 'blue', 'green', 'green', 'brown']
overlay = [ False, True, True, False, False, True, True, True, True, False]
scatter = [ False, False, False, False, False, True, False, False, False, True]

from utilities import limit_order_duration_on_exchange, get_data
def SIGNAL():
    return dfpl.total_signal

class MyStrat(Strategy):
    initsize = 0.99
    mysize = initsize

    def init(self):
        super().init()
        _LOG.debug(f"initiating backtest")
        # for idx, p in enumerate(indicators):
        #     setattr(self, p,
        #             self.I(get_data, self.data, p, color=colors[idx], scatter=scatter[idx], overlay=overlay[idx]))
        # self.signal = self.I(SIGNAL)
        a=1

    def next(self):
        super().next()

        # slatr = 1.2 * self.ATR[-1]
        TPSLRatio = 1.5
        perc = 0.02
        today_idx = self.data._Data__i -1
        today = self.data.index[-1]
        _LOG.debug(f"current candle {today_idx}")
        price_close = self.data.Close[-1]

        def is_long_close_signal():
            return True if self.data.RSI[-1] >= 75 else False

        def is_short_close_signal():
            return True if self.data.RSI[-1] <= 25 else False

        def is_long_buy_signal():
            return True if self.data.total_signal[-1] != 0 else False

        def is_short_sell_signal():
            return True if self.data.total_signal[-1] != 0 else False

        # days_max_order_to_fulfill = 10
        # check_duration_on_exchange(self, days_max_order_to_fulfill, today_idx)
        # if len(self.trades) > 0:
        #     if self.data.index[-1] - self.trades[-1].entry_time >= 10:
        #         self.trades[-1].close()
        #     if self.trades[-1].is_long and is_long_close_signal():
        #         # _LOG.debug(f"closing long trade nr: {_trade}, entrybar {trade.entry_bar}")
        #         self.trades[-1].close()
        #     elif self.trades[-1].is_short and is_short_close_signal():
        #         # _LOG.debug(f"closing short trade nr: {_trade}, entrybar {trade.entry_bar}")
        #         self.trades[-1].close()

        # determine if close trades
        _trade = 0
        if len(self.trades) > 0:
            _LOG.debug(f"total open trades: {len(self.trades)}")
            for trade in self.trades:
                _trade += 1
                _LOG.debug(f"trade nr: {_trade} entrybar {trade.entry_bar}")
                # if today - trade.entry_time >= pd.Timedelta(10, unit="d"):
                # if today - trade.entry_time >= 10:
                if today_idx - trade.tag >= 10:
                    trade.close()
                if trade.is_long and is_long_close_signal():
                    # _LOG.debug(f"closing long trade nr: {_trade}, entrybar {trade.entry_bar}")
                    trade.close()
                elif trade.is_short and is_short_close_signal():
                    # _LOG.debug(f"closing short trade nr: {_trade}, entrybar {trade.entry_bar}")
                    trade.close()

        buyprice = self.data.total_signal[-1]
        if is_long_buy_signal() and len(self.trades) == 0 and self.data['ema200_signal']  == 2:
            #Cancel previous orders
            # for order in self.orders:
            #     order.cancel()
            # sl1 = price_close - slatr
            # tp1 = price_close + slatr * TPSLRatio
            sl1 = min(self.data.Low[-1], self.data.Low[-2]) * (1 - perc)
            tp1 = price_close + (price_close - sl1) * TPSLRatio
            self.buy(sl=sl1, tp=tp1, size=self.mysize, tag=today_idx+1)
            _LOG.debug(f"buyorder at bar: {today_idx}, price: {buyprice}")
            # self.buy(sl=buyprice / 2, limit=buyprice, size=self.initsize, tag=today_idx)

        elif is_short_sell_signal() and len(self.trades) == 0 and self.data['ema200_signal']  == 1:
            #Cancel previous orders
            # for order in self.orders:
            #     order.cancel()
            # sl1 = price_close + slatr
            # tp1 = price_close - slatr * TPSLRatio
            # self.sell(sl=sl1, tp=tp1, size=self.mysize,tag=today_idx)
            sl1 = max(self.data.High[-1], self.data.High[-2]) * (1 + perc)
            tp1 = price_close - (sl1 - price_close) * TPSLRatio
            self.sell(sl=sl1, tp=tp1, size=self.mysize, tag=today_idx+1)
            _LOG.debug(f"buyorder at bar: {today_idx}, price: {buyprice}")
            # self.sell(sl=buyprice * 2, limit=buyprice, size=self.initsize, tag=today_idx)


bt = Backtest(dfpl, MyStrat, cash=1000, margin=1 / 5, commission=0.00)
stat = bt.run()
print(stat)
print(stat._trades)
# bt.plot(plot_width=1500)
bt.plot(plot_width=1500, smooth_equity=True, superimpose=True, reverse_indicators=False)
# update plot with signals from backtest
date_strt, date_end = dfpl_plot.index[0], dfpl_plot.index[-1]
res = False
_trades_entry = []
for idx, ele in enumerate(stat._trades['EntryTime']):
    # checking for date in range
    if ele >= date_strt and ele <= date_end:
        _trades_entry.append(idx)
stat_cp = stat.copy()
a_trades = stat_cp._trades.iloc[_trades_entry]
metric_figure, fig, bottom_figure = update_plots_with_signals(a_trades, metric_figure, fig, bottom_figure)
metric_figure.show()
