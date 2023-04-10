import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
# from fin import fetch_download_data
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from backtesting import Strategy
from backtesting import Backtest
from backtesting.test import GOOG
from backtesting.lib import crossover
import logging
import os

_LOG = logging.getLogger(__name__)
# run this to start logging
logging.debug("Begin")
_LOG.setLevel(logging.DEBUG)

stck = '^GSPC'
# stck = '^RUI'
# stck = 'EURUSD=X'
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Project root
# CSV_DIR = os.path.join(ROOT_DIR, 'backtesting', 'examples')

df=pd.read_csv(os.path.join(ROOT_DIR, "EURUSD_Candlestick_5_M_ASK_30.09.2019-30.09.2022.csv"))
df=df[0:50000]
df["Gmt time"]=df["Gmt time"].str.replace(".000","")
df['Gmt time']=pd.to_datetime(df['Gmt time'],format='%d.%m.%Y %H:%M:%S')
df.set_index("Gmt time", inplace=True)
df=df[df.High!=df.Low]
len(df)
df = df

df["VWAP"]=ta.vwap(df.High, df.Low, df.Close, df.Volume)
df['RSI']=ta.rsi(df.Close, length=16)
my_bbands = ta.bbands(df.Close, length=14, std=2.0)
df=df.join(my_bbands)
vwap_sig = [0]*len(df)
backcandles = 15

for row in range(backcandles, len(df)):
    upt = 1
    dnt = 1
    for i in range(row-backcandles, row+1):
        if max(df.Open[i], df.Close[i])>=df.VWAP[i]:
            dnt=0
        if min(df.Open[i], df.Close[i])<=df.VWAP[i]:
            upt=0
    if upt==1 and dnt==1:
        vwap_sig[row]=3
    elif upt==1:
        vwap_sig[row]=2
    elif dnt==1:
        vwap_sig[row]=1

df['vwap_sig'] = vwap_sig


def TotalSignal(l):
    if (df.vwap_sig[l] == 2
            and df.Close[l] <= df['BBL_14_2.0'][l]
            and df.RSI[l] < 45):
        return 2
    if (df.vwap_sig[l] == 1
            and df.Close[l] >= df['BBU_14_2.0'][l]
            and df.RSI[l] > 55):
        return 1
    return 0


TotSignal = [0] * len(df)
for row in range(backcandles, len(df)):  # careful backcandles used previous cell
    TotSignal[row] = TotalSignal(row)
df['TotalSignal'] = TotSignal

df[df.TotalSignal!=0].count()


def pointposbreak(x):
    if x['TotalSignal']==1:
        return x['High']+1e-4
    elif x['TotalSignal']==2:
        return x['Low']-1e-4
    else:
        return np.nan

df['pointposbreak'] = df.apply(lambda row: pointposbreak(row), axis=1)


st=10400
dfpl = df[st:st+350]
dfpl.reset_index(inplace=True)
fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                open=dfpl['Open'],
                high=dfpl['High'],
                low=dfpl['Low'],
                close=dfpl['Close']),
                go.Scatter(x=dfpl.index, y=dfpl.VWAP,
                           line=dict(color='blue', width=1),
                           name="VWAP"),
                go.Scatter(x=dfpl.index, y=dfpl['BBL_14_2.0'],
                           line=dict(color='green', width=1),
                           name="BBL"),
                go.Scatter(x=dfpl.index, y=dfpl['BBU_14_2.0'],
                           line=dict(color='green', width=1),
                           name="BBU")])

fig.add_scatter(x=dfpl.index, y=dfpl['pointposbreak'], mode="markers",
                marker=dict(size=10, color="MediumPurple"),
                name="Signal")
fig.show()

dfpl = df[:75000].copy()

dfpl['ATR']=ta.atr(dfpl.High, dfpl.Low, dfpl.Close, length=7)
#help(ta.atr)

def SIGNAL():
    return dfpl.TotalSignal



class MyStrat(Strategy):
    initsize = 0.99
    mysize = initsize

    def init(self):
        super().init()
        self.signal1 = self.I(SIGNAL)

    def next(self):
        super().next()
        slatr = 1.2 * self.data.ATR[-1]
        TPSLRatio = 1.5

        if len(self.trades) > 0:
            if self.trades[-1].is_long and self.data.RSI[-1] >= 90:
                self.trades[-1].close()
            elif self.trades[-1].is_short and self.data.RSI[-1] <= 10:
                self.trades[-1].close()

        if self.signal1 == 2 and len(self.trades) == 0:
            sl1 = self.data.Close[-1] - slatr
            tp1 = self.data.Close[-1] + slatr * TPSLRatio
            self.buy(sl=sl1, tp=tp1, size=self.mysize)

        elif self.signal1 == 1 and len(self.trades) == 0:
            sl1 = self.data.Close[-1] + slatr
            tp1 = self.data.Close[-1] - slatr * TPSLRatio
            self.sell(sl=sl1, tp=tp1, size=self.mysize)


bt = Backtest(dfpl, MyStrat, cash=100, margin=1 / 10, commission=0.00)
stat = bt.run()
print(stat)
bt.plot(plot_width=1500,show_legend=False)
# now = datetime.now()
# now_minus_10 = now + relativedelta(years=-10)
# # get ticker data for the last 10  years
# dfSPY = fetch_download_data(now_minus_10, now, stck)

# dfSPY = GOOG
# dfSPY=yf.download("^GSPC",start='2011-01-05', end='2021-01-05')
# #dfSPY = pd.read_csv("SPY.USUSD_Candlestick_1_D_BID_16.02.2017-21.05.2022.csv")
# # dfSPY=yf.download("^GSPC",start=now_minus_10, end=now')
# #dfSPY=yf.download("^RUI",start='2011-01-05', end='2021-01-05')
# #dfSPY=yf.download("EURUSD=X",start='2011-01-05', end='2021-01-05')
#
# dfSPY=dfSPY[dfSPY.High!=dfSPY.Low]
# dfSPY.reset_index(inplace=True)
# dfSPY.head()

#
# #
# dfSPY['EMA200']=ta.sma(dfSPY.Close, length=200)#sma ema
# dfSPY['EMA20']=ta.sma(dfSPY.Close, length=20)#sma ema
# dfSPY['RSI']=ta.rsi(dfSPY.Close, length=2)
# # KST osciallator (momentum)
# dfSPY_KST=ta.kst(dfSPY.Close, 10,15,20,30,10,10,10,15,9)
# dfSPY=dfSPY.join(dfSPY_KST)
# #dfSPY.ta.indicators()
# # #help(ta.bbands)
# my_bbands = ta.bbands(dfSPY.Close, length=20, std=2.5)
# # print(my_bbands[0:50])
# dfSPY=dfSPY.join(my_bbands)
#
# # drop first days so that we have good data from day 1 (200ma etc)
# n = 220
# dfSPY.drop(index=dfSPY.index[:n],inplace=True)
# print(dfSPY)
#
#
# class MStrategy(Strategy):
#     initsize = 0.9999
#     upper_bound = 70
#     lower_bound = 30
#     rsi_window = 14
#     trend_200 = []
#
#     ordertime_idx = []
#
#
#     def init(self, *args):
#         super().init()
#
#         def get_data(data, column_name):
#             if not hasattr(data, column_name):
#                 _LOG.error(f'indicator does not exist in dataframe {column_name}')
#             return data[column_name]
#
#         # precalculated data
#         indicators = ['EMA200', 'EMA20', 'BBL_20_2.5', 'BBU_20_2.5']
#         colors = [None, None, 'red', 'red']
#
#         for idx, p in enumerate(indicators):
#             setattr(self, p, self.I(get_data, self.data, p, color=colors[idx], overlay=True))
#
#     def next(self):
#         super().next()
#         price = self.data.Close[-1]
#         today = self.data.index[-1]
#         today_idx = self.data._Data__i
#
#         buyprice = price * 0.90
#         # only allow orders to be a given days on stock before closed
#         days_max_order_to_fulfill = 1
#         for j, order in enumerate(self.orders):
#             # we ignore the order if it is fulfilled and a trade is created
#             # is contingent means it has been fulfilled
#             if order.is_contingent:
#                 #remove the fulfilled orders from the list
#                 pass
#             # if its not fulfilled, check how long it has been on the exchange
#             else:
#                 # allow the order to stay on exchange for given days before its cancelled
#                 if today_idx - order.tag >= days_max_order_to_fulfill:
#                     _LOG.debug(f'non-fulfilled limit order at {order.limit} expires now: {order.tag} after {today_idx - order.tag} days')
#                     order.cancel()
#                     # self.ordertime_idx.pop(j)
#                     _LOG.debug(f"total orders: {len(self.orders)}")
#                     break
#
#         _trade = 0
#         if len(self.trades) > 0:
#             _LOG.debug(f"total open trades: {len(self.trades)}")
#             for trade in self.trades:
#                 _trade += 1
#                 # if today_idx - trade.entry_bar >= 1000:
#                 #     trade.close()
#
#                 # if trade.is_long and self.data.RSI[-1] >= 50:
#                 #     trade.close()
#                 # elif trade.is_short and self.data.RSI[-1] <= 50:
#                 #     trade.close()
#                 _LOG.debug(f"trade nr: {_trade} entrybar {trade.entry_bar}")
#                 if trade.is_long and not self.trend_200:
#                     _LOG.debug(f"closing long trade nr: {_trade}, entrybar {trade.entry_bar}")
#                     trade.close()
#                 elif trade.is_short and self.trend_200:
#                     _LOG.debug(f"closing short trade nr: {_trade}, entrybar {trade.entry_bar}")
#                     trade.close()
#
#         if price >= self.data.EMA200[-1]:
#             self.trend_200 = True # updtrend
#         else:
#             self.trend_200 = False
#
#         if self.trend_200 and len(self.trades) == 0:
#             self.buy(sl=buyprice / 5, limit=buyprice, size=self.initsize, tag=today_idx)
#             _LOG.debug(f"buyorder at bar: {today_idx}, price: {buyprice}")
#             self.ordertime_idx.append(today_idx)
#
#
# bt = Backtest(dfSPY, MStrategy, cash=10000, margin=1, commission=.00)
# stat = bt.run()
# print(stat)
# print(stat._trades)
#
# bt.plot(plot_width=1500,smooth_equity=True, superimpose=True,reverse_indicators=False)
#
# dfpl = dfSPY.copy()
# # dfpl = dfSPY[1000:1500].copy()
# #dfpl=dfpl.drop(columns=['level_0'])#!!!!!!!!!!
# #dfpl.reset_index(inplace=True)
# fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
#                 open=dfpl['Open'],
#                 high=dfpl['High'],
#                 low=dfpl['Low'],
#                 close=dfpl['Close']),
#                 go.Scatter(x=dfpl.index, y=dfpl.EMA200, line=dict(color='orange', width=2), name="EMA200"),
#                 go.Scatter(x=dfpl.index, y=dfpl.EMA20, line=dict(color='yellow', width=2), name="EMA20"),
#                 # go.Scatter(x=dfpl.index, y=dfpl.KST_10_15_20_30_10_10_10_15, line=dict(color='green', width=2), name="KST_10_15_20_30_10_10_10_15"),
#                 # go.Scatter(x=dfpl.index, y=dfpl.KSTs_9, line=dict(color='black', width=2), name="KSTs_9"),
#                 #go.Scatter(x=dfpl.index, y=dfpl['BBL_20_2.5'], line=dict(color='blue', width=1), name="BBL_20_2.5"),
#                 #go.Scatter(x=dfpl.index, y=dfpl['BBU_20_2.5'], line=dict(color='blue', width=1), name="BBU_20_2.5")
#                       ])
#
# fig.add_scatter(x=stat._trades.EntryTime, y=stat._trades.EntryPrice, mode="markers",
#                 marker=dict(size=6, color="Black"),
#                 name="Entrybar")
# fig.add_scatter(x=stat._trades.ExitTime, y=stat._trades.ExitPrice, mode="markers",
#                 marker=dict(size=6, color="Red"),
#                 name="Exitbar")
#
# # fig.add_scatter(x=dfpl.index, y=stat._trades.ExitBar, mode="markers",
# #                 marker=dict(size=6, color="MediumPurple"),
# #                 name="ExitBar")
# # fig.update(layout_yaxis_range = [300,420])
# # fig.update_xaxes(rangeslider_visible=False)
# fig.update_layout(autosize=False, width=1500, height=1000,margin=dict(l=50,r=50,b=100,t=100,pad=4), paper_bgcolor="white")
# fig.show()