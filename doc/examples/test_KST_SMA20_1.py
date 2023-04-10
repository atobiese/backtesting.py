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

_LOG = logging.getLogger(__name__)
# run this to start logging
logging.debug("Begin")
_LOG.setLevel(logging.DEBUG)

stck = '^GSPC'
# stck = '^RUI'
# stck = 'EURUSD=X'

# now = datetime.now()
# now_minus_10 = now + relativedelta(years=-10)
# # get ticker data for the last 10  years
# dfSPY = fetch_download_data(now_minus_10, now, stck)

dfSPY = GOOG
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
dfSPY['EMA200']=ta.sma(dfSPY.Close, length=200)#sma ema
dfSPY['EMA20']=ta.sma(dfSPY.Close, length=20)#sma ema
dfSPY['RSI']=ta.rsi(dfSPY.Close, length=2)
# KST osciallator (momentum)
dfSPY_KST=ta.kst(dfSPY.Close, 10,15,20,30,10,10,10,15,9)
dfSPY=dfSPY.join(dfSPY_KST)
#dfSPY.ta.indicators()
# #help(ta.bbands)
my_bbands = ta.bbands(dfSPY.Close, length=20, std=2.5)
# print(my_bbands[0:50])
dfSPY=dfSPY.join(my_bbands)

# drop first days so that we have good data from day 1 (200ma etc)
n = 220
dfSPY.drop(index=dfSPY.index[:n],inplace=True)
print(dfSPY)

from utilities import limit_order_duration_on_exchange, get_data

class MStrategy(Strategy):
    initsize = 0.9999
    upper_bound = 70
    lower_bound = 30
    rsi_window = 14
    trend_200 = []

    ordertime_idx = []


    def init(self, *args):
        super().init()

        # precalculated data
        indicators = ['EMA200', 'EMA20', 'BBL_20_2.5', 'BBU_20_2.5']
        colors = [None, None, 'red', 'red']

        for idx, p in enumerate(indicators):
            setattr(self, p, self.I(get_data, self.data, p, color=colors[idx], overlay=True))

    def next(self):
        super().next()
        price = self.data.Close[-1]
        today = self.data.index[-1]
        today_idx = self.data._Data__i

        buyprice = price * 0.90
        # only allow orders to be a given days on stock before closed
        days_max_order_to_fulfill = 1
        limit_order_duration_on_exchange(self, days_max_order_to_fulfill, today_idx)

        _trade = 0
        if len(self.trades) > 0:
            _LOG.debug(f"total open trades: {len(self.trades)}")
            for trade in self.trades:
                _trade += 1
                # if today_idx - trade.entry_bar >= 1000:
                #     trade.close()

                # if trade.is_long and self.data.RSI[-1] >= 50:
                #     trade.close()
                # elif trade.is_short and self.data.RSI[-1] <= 50:
                #     trade.close()
                _LOG.debug(f"trade nr: {_trade} entrybar {trade.entry_bar}")
                if trade.is_long and not self.trend_200:
                    _LOG.debug(f"closing long trade nr: {_trade}, entrybar {trade.entry_bar}")
                    trade.close()
                elif trade.is_short and self.trend_200:
                    _LOG.debug(f"closing short trade nr: {_trade}, entrybar {trade.entry_bar}")
                    trade.close()

        if price >= self.data.EMA200[-1]:
            self.trend_200 = True # updtrend
        else:
            self.trend_200 = False

        if self.trend_200 and len(self.trades) == 0:
            self.buy(sl=buyprice / 5, limit=buyprice, size=self.initsize, tag=today_idx)
            _LOG.debug(f"buyorder at bar: {today_idx}, price: {buyprice}")
            self.ordertime_idx.append(today_idx)


bt = Backtest(dfSPY, MStrategy, cash=10000, margin=1, commission=.00)
stat = bt.run()
print(stat)
print(stat._trades)

bt.plot(plot_width=1500,smooth_equity=True, superimpose=True,reverse_indicators=False)

dfpl = dfSPY.copy()
# dfpl = dfSPY[1000:1500].copy()
#dfpl=dfpl.drop(columns=['level_0'])#!!!!!!!!!!
#dfpl.reset_index(inplace=True)
fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                open=dfpl['Open'],
                high=dfpl['High'],
                low=dfpl['Low'],
                close=dfpl['Close']),
                go.Scatter(x=dfpl.index, y=dfpl.EMA200, line=dict(color='orange', width=2), name="EMA200"),
                go.Scatter(x=dfpl.index, y=dfpl.EMA20, line=dict(color='yellow', width=2), name="EMA20"),
                # go.Scatter(x=dfpl.index, y=dfpl.KST_10_15_20_30_10_10_10_15, line=dict(color='green', width=2), name="KST_10_15_20_30_10_10_10_15"),
                # go.Scatter(x=dfpl.index, y=dfpl.KSTs_9, line=dict(color='black', width=2), name="KSTs_9"),
                #go.Scatter(x=dfpl.index, y=dfpl['BBL_20_2.5'], line=dict(color='blue', width=1), name="BBL_20_2.5"),
                #go.Scatter(x=dfpl.index, y=dfpl['BBU_20_2.5'], line=dict(color='blue', width=1), name="BBU_20_2.5")
                      ])

fig.add_scatter(x=stat._trades.EntryTime, y=stat._trades.EntryPrice, mode="markers",
                marker=dict(size=6, color="Black"),
                name="Entrybar")
fig.add_scatter(x=stat._trades.ExitTime, y=stat._trades.ExitPrice, mode="markers",
                marker=dict(size=6, color="Red"),
                name="Exitbar")

# fig.add_scatter(x=dfpl.index, y=stat._trades.ExitBar, mode="markers",
#                 marker=dict(size=6, color="MediumPurple"),
#                 name="ExitBar")
# fig.update(layout_yaxis_range = [300,420])
# fig.update_xaxes(rangeslider_visible=False)
fig.update_layout(autosize=False, width=1500, height=1000,margin=dict(l=50,r=50,b=100,t=100,pad=4), paper_bgcolor="white")
fig.show()