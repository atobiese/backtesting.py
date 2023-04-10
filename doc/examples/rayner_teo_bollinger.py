import pandas as pd
import yfinance as yf

# #dfSPY=yf.download("^GSPC",start='2011-01-05', end='2021-01-05')
# dfSPY=yf.download("^RUI",start='2011-01-05', end='2021-01-05')
# #dfSPY=yf.download("EURUSD=X",start='2011-01-05', end='2021-01-05')
#
# dfSPY=dfSPY[dfSPY.High!=dfSPY.Low]
# dfSPY.reset_index(inplace=True)
# dfSPY.head()
dfSPY = yf.download("^RUI", start='2011-01-05', end='2021-01-05')
dfSPY = dfSPY[dfSPY.High != dfSPY.Low]
n = 0
dfSPY.drop(index=dfSPY.index[:n], inplace=True)
dfSPY.reset_index(inplace=True)


import pandas_ta as ta
dfSPY['EMA']=ta.ema(dfSPY.Close, length=200)#sma ema
# dfSPY['EMA2']=ta.ema(dfSPY.Close, length=150)#sma ema
dfSPY['RSI']=ta.rsi(dfSPY.Close, length=12)

my_bbands = ta.bbands(dfSPY.Close, length=14, std=2.0)

dfSPY=dfSPY.join(my_bbands)
dfSPY.dropna(inplace=True)
dfSPY.reset_index(inplace=True)


def addemasignal(df, backcandles):
    emasignal = [0]*len(df)
    for row in range(backcandles, len(df)):
        upt = 1
        dnt = 1
        for i in range(row-backcandles, row+1):
            if df.High[i]>=df.EMA[i]:
                dnt=0
            if df.Low[i]<=df.EMA[i]:
                upt=0
        if upt==1 and dnt==1:
            #print("!!!!! check trend loop !!!!")
            emasignal[row]=3
        elif upt==1:
            emasignal[row]=2
        elif dnt==1:
            emasignal[row]=1
    df['EMASignal'] = emasignal

# def addemasignal(df):
#     emasignal = [0]*len(df)
#     for i in range(0, len(df)):
#         if df.EMA2[i]>df.EMA[i]:
#             emasignal[i]=2
#         elif df.EMA2[i]<df.EMA[i]:
#             emasignal[i]=1
#     df['EMASignal'] = emasignal
addemasignal(dfSPY, 6)


def addorderslimit(df, percent):
    ordersignal = [0] * len(df)
    for i in range(1, len(df)):  # EMASignal of previous candle!!! modified!!!
        if df.Close[i] <= df['BBL_14_2.0'][i] and df.EMASignal[i] == 2:
            ordersignal[i] = df.Close[i] - df.Close[i] * percent
        elif df.Close[i] >= df['BBU_14_2.0'][i] and df.EMASignal[i] == 1:
            ordersignal[i] = df.Close[i] + df.Close[i] * percent
    df['ordersignal'] = ordersignal


addorderslimit(dfSPY, 0.000)

dfSPY[dfSPY.ordersignal!=0]

"""# Visualization"""

import numpy as np
def pointposbreak(x):
    if x['ordersignal']!=0:
        return x['ordersignal']
    else:
        return np.nan
dfSPY['pointposbreak'] = dfSPY.apply(lambda row: pointposbreak(row), axis=1)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime



dfpl = dfSPY[:].copy()
def SIGNAL():
    return dfpl.ordersignal


from backtesting import Strategy
from backtesting import Backtest


class MyStrat(Strategy):
    initsize = 0.99
    mysize = initsize

    def init(self):
        super().init()
        # self.signal = self.I(SIGNAL)

    def next(self):
        super().next()
        TPSLRatio = 1.5
        perc = 0.02

        if len(self.trades) > 0:
            if self.data.index[-1] - self.trades[-1].entry_time >= 10:
                self.trades[-1].close()
            if self.trades[-1].is_long and self.data.RSI[-1] >= 75:
                self.trades[-1].close()
            elif self.trades[-1].is_short and self.data.RSI[-1] <= 25:
                self.trades[-1].close()

        if self.data.ordersignal != 0 and len(self.trades) == 0 and self.data.EMASignal == 2:
            sl1 = min(self.data.Low[-1], self.data.Low[-2]) * (1 - perc)
            tp1 = self.data.Close[-1] + (self.data.Close[-1] - sl1) * TPSLRatio
            self.buy(sl=sl1, tp=tp1, size=self.mysize)

        elif self.data.ordersignal != 0 and len(self.trades) == 0 and self.data.EMASignal == 1:
            sl1 = max(self.data.High[-1], self.data.High[-2]) * (1 + perc)
            tp1 = self.data.Close[-1] - (sl1 - self.data.Close[-1]) * TPSLRatio
            self.sell(sl=sl1, tp=tp1, size=self.mysize)


bt = Backtest(dfpl, MyStrat, cash=1000, margin=1 / 5, commission=.000)
stat = bt.run()
print(stat)

# bt.plot()

