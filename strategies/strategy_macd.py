from backtesting import Strategy
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
# from fin import fetch_download_data
from backtesting.test import GOOG
from backtesting.lib import crossover, cross
import logging
import os

from doc.examples.utilities import  plot_visualization, update_plots_with_signals
from doc.examples.utilities import  limit_order_duration_on_exchange, get_data, fetch_download_data

import logging

_LOG = logging.getLogger(__name__)
# run this to start logging
logging.debug("Begin")
_LOG.setLevel(logging.DEBUG)

''' main strategy file to run backtests 
 rules:
 
 
'''


def get_signals():
    dummy_data = False
    if not dummy_data:
        load_from_csv = True
        # stck = '^GSPC'
        # stck = '^RUI'
        # stck = 'EURUSD=X'
        stck = 'AAPL'
        # now = datetime.now()
        # start = now + relativedelta(years=-10)
        # end = datetime.date.today()

        start = datetime.date(2018, 1, 2)
        end = datetime.date(2021, 4, 16+1)

        # # get ticker data for the last 10  years
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Project root
        CSV_DIR = os.path.join(ROOT_DIR, "../doc/examples/temp_csv")
        df = fetch_download_data(start, end, stck, load_from_csv=load_from_csv, csv_directory=CSV_DIR)
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
    # df['EMA200'] = ta.ema(df.Close, length=200)  # sma ema
    # df['EMA3'] = ta.ema(df.Close, length=3)
    # # df["VWAP"] = ta.vwap(df.High, df.Low, df.Close, df.Volume)
    # df['RSI'] = ta.rsi(df.Close, length=12)
    # my_bbands = ta.bbands(df.Close, length=14, std=2.0)
    # df = df.join(my_bbands)
    _df_macd = ta.macd(df.Close, 12, 26, 9)
    df = df.join(_df_macd)
    # df['ATR'] = ta.atr(df.High, df.Low, df.Close, length=7)
    # # KST osciallator (momentum)
    # _df_KST = ta.kst(df.Close, 10, 15, 20, 30, 10, 10, 10, 15, 9)
    # df = df.join(_df_KST)

    # df['EMA20'] = ta.ema(df.Close, length=20)  # sma ema
    # backrollingN = 2
    # slope_mea = df['EMA20'].diff(periods=1)
    # df['slope_ema20'] = slope_mea.rolling(window=backrollingN).mean()

    # df['KST_10_15_20_30_10_10_10_15']
    # df['KSTs_9']
    # strategy:
    #
    # def addemasignal(df, backcandles, ema_name):
    #     emasignal = [0] * len(df)
    #     for row in range(backcandles, len(df)):
    #         upt = 1
    #         dnt = 1
    #         for i in range(row - backcandles, row + 1):
    #             if df.High[i] >= df[ema_name][i]:
    #                 dnt = 0
    #             if df.Low[i] <= df[ema_name][i]:
    #                 upt = 0
    #         if upt == 1 and dnt == 1:
    #             # print("!!!!! check trend loop !!!!")
    #             emasignal[row] = 3
    #         elif upt == 1:
    #             emasignal[row] = 2
    #         elif dnt == 1:
    #             emasignal[row] = 1
    #     # singal name
    #     signal_name = f'{ema_name.lower()}_signal'
    #     df[signal_name] = emasignal

    # nr_for_confirmation = 1
    # addemasignal(df, nr_for_confirmation, 'EMA200')
    #
    # nr_for_confirmation = 1
    # addemasignal(df, nr_for_confirmation, 'EMA20')

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

    def _total_signal_next(l):
        if (
                # crossover(df['MACD_12_26_9'][l - 2:l], df['MACDs_12_26_9'][l - 2:l])
                df['MACDs_12_26_9'][l] < df['MACD_12_26_9'][l]
                # df.EMA200[l] < df.Close[l]
                # df.ema20_signal[l] == 2
                # and df.ema20_signal[l] == 2
                # df.EMA3[l] - df.EMA20[l] >= 0
                # df['slope_ema20'][l] > 0
                # df['KSTs_9'][l] <= df['KST_10_15_20_30_10_10_10_15'][l]):
                # and crossover(df['KST_10_15_20_30_10_10_10_15'][l - 2:l], df['KSTs_9'][l - 2:l])):
                # and df.RSI[l] < 45
                ):
            a = 1
            return 2
        if (
                # crossover( df['MACDs_12_26_9'][l - 2:l],df['MACD_12_26_9'][l - 2:l])
                # df.EMA200[l] > df.Close[l]
                #df.ema200_signal[l] == 1
                # and df.ema20_signal[l] == 1
                # and df.Close[l] >= df['BBU_14_2.0'][l]
                df['MACDs_12_26_9'][l] > df['MACD_12_26_9'][l]
                # and df['KSTs_9'][l] >= df['KST_10_15_20_30_10_10_10_15'][l]):
                # current and previous candle
                # df['slope_ema20'][l] <= 0
                # two candle confirmation crossover
                # and crossover(df['KSTs_9'][l - 2:l], df['KST_10_15_20_30_10_10_10_15'][l - 2:l])):
            # and df.RSI[l] > 55
            ):
            return 1
        return 0

    # def addorderslimit(df, percent):
    #     ordersignal = [0] * len(df)
    #     for i in range(1, len(df)):  # EMASignal of previous candle!!! modified!!!
    #         if df.Close[i] <= df['BBL_14_2.0'][i] and df.EMASignal[i] == 2:
    #             ordersignal[i] = df.Close[i] - df.Close[i] * percent
    #         elif df.Close[i] >= df['BBU_14_2.0'][i] and df.EMASignal[i] == 1:
    #             ordersignal[i] = df.Close[i] + df.Close[i] * percent
    #     df['ordersignal'] = ordersignal
    #
    # addorderslimit(df, 0.000)

    backcandles = 1
    nr_samples = len(df)
    total_sig = [0] * len(df)
    for row in range(backcandles, nr_samples):  # careful backcandles used previous cell
        total_sig[row] = _total_signal_next(row)
    df['total_signal'] = total_sig

    def pointposbreak(x):
        if x['total_signal'] == 2:
            return x['Low'] - x['Low']*0.01


    def pointnegbreak(x):
        if x['total_signal'] == 1:
            return x['High'] + x['Low']*0.01

    df['pointposbreak'] = df.apply(lambda row: pointposbreak(row), axis=1)
    df['pointnegbreak'] = df.apply(lambda row: pointnegbreak(row), axis=1)

    data_g = {
        "initsize": 0.9999,
        "TPSLRatio": 1.5,
        "perc": 0.02,
        "days_max_order_to_fulfill": 5
    }

    # display
    # indicators = ['EMA200', 'BBL_14_2.0', 'BBU_14_2.0', 'BBM_14_2.0', 'total_signal']
    # colors = ['MediumPurple', 'orange', 'orange', 'black', 'black']
    indicators0 = ['pointposbreak', 'pointnegbreak']
    scatter0 = [True, True]
    colors0 = ['green', 'red']

    sub_indicators0 = ['MACD_12_26_9', 'MACDs_12_26_9', 'total_signal']
    sub_colors0 = ['black', 'green', 'black', 'green']
    sub2_ind0 = ['total_signal']
    sub2_col0 = ['black']


    # for bactest.py
    indicators =['MACD_12_26_9', 'MACDs_12_26_9', 'total_signal', 'pointposbreak']
    colors = ['black', 'green', 'blue', 'green']
    overlay = [False, False, True, False, True]
    scatter = [False, False, False, True]

    # run test
    init_data_g = {
        "cash": 1000,
        "margin": 1.0,
        "commission": 0.00495,
        "exclusive_orders": True,
        "trade_on_close": True,
    }


    return df, data_g, indicators0, colors0, \
           scatter0, sub_indicators0, sub_colors0, \
           colors, overlay, indicators, scatter, sub2_ind0, sub2_col0, \
           init_data_g


def create_buy_order(self):
    if len(self.trades) == 0:
        self.buy()
    # # Cancel previous orders
    # # for order in self.orders:
    # #     order.cancel()
    # # sl1 = price_close - slatr
    # # tp1 = price_close + slatr * TPSLRatio
    # TPSLRatio = self.data_g['TPSLRatio']
    # perc = self.data_g['perc']
    #
    # today_idx = self.data._Data__i - 1
    # today = self.data.index[-1]
    # _LOG.debug(f"current candle {today_idx}")
    # price_close = self.data.Close[-1]
    #
    #
    # percent = 0.03
    # price_limit = self.data.Close[-1] - (self.data.Close[-1] * percent)
    #
    # # Cancel previous order(s) as we only want one order at a time
    # # decied to close previous and open a new order
    # keep_previous_open_orders = True
    # if not keep_previous_open_orders:
    #     for order in self.orders:
    #         order.cancel()
    #     # sl1 = min(self.data.Low[-1], self.data.Low[-2]) * (1 - perc)
    #     # tp1 = price_close + (price_close - sl1) * TPSLRatio
    #     # self.buy(sl=tp1/2, tp=tp1, size=self.mysize, tag=today_idx + 1)
    #     _LOG.debug(f"buyorder at bar: {today_idx}, limitprice: {price_limit}, close price: {price_close}")
    #     self.buy(sl=price_close / 2, limit=price_close, size=self.mysize, tag=today_idx)
    # else:
    #     if len(self.orders) > 0:
    #         return
    #     else:
    #         _LOG.debug(f"buyorder at bar: {today_idx}, limitprice: {price_limit}, close price: {price_close}")
    #         self.buy(sl=price_close / 2, limit=price_close, size=self.mysize, tag=today_idx)


# def create_sell_order(self):
#     self.position.close()
#     # # Cancel previous orders
#     # # for order in self.orders:
#     # #     order.cancel()
#     # # sl1 = price_close + slatr
#     # # tp1 = price_close - slatr * TPSLRatio
#     # # self.sell(sl=sl1, tp=tp1, size=self.mysize,tag=today_idx)
#     # TPSLRatio = self.data_g['TPSLRatio']
#     # perc = self.data_g['perc']
#     #
#     # today_idx = self.data._Data__i - 1
#     # today = self.data.index[-1]
#     # _LOG.debug(f"current candle {today_idx}")
#     # price_close = self.data.Close[-1]
#     #
#     # percent = 0.03
#     # price_limit = self.data.Close[-1] - (self.data.Close[-1] * percent)
#     #
#     # # sl1 = max(self.data.High[-1], self.data.High[-2]) * (1 + perc)
#     # # tp1 = price_close - (sl1 - price_close) * TPSLRatio
#     # # self.sell(sl=tp1/2, tp=tp1, size=self.mysize, tag=today_idx + 1)
#     # _LOG.debug(f"buyorder at bar: {today_idx}, limitprice: {price_limit}, close price: {price_close}")
#     # self.buy(sl=price_close / 2, limit=price_close, size=self.mysize, tag=today_idx)


def create_close_on_trades(self):
    today_idx = self.data._Data__i - 1
    if len(self.trades) > 0:
        _LOG.debug(f"total open trades: {len(self.trades)}")
        _trade = 0
        # _days_to_close_on_excahnge = 10
        for trade in self.trades:
            _trade += 1
            _LOG.debug(f"trade nr: {_trade} entrybar {trade.entry_bar}")
            # if today - trade.entry_time >= pd.Timedelta(10, unit="d"):
            # if today - trade.entry_time >= 10:
            # if today_idx - trade.tag >= _days_to_close_on_excahnge:
            #     trade.close()
            if trade.is_long and is_long_close_signal(self):
                _LOG.debug(f"closing long trade nr: {_trade}, entrybar {trade.entry_bar}")
                trade.close()
            # elif trade.is_short and is_short_close_signal(self):
            #     _LOG.debug(f"closing short trade nr: {_trade}, entrybar {trade.entry_bar}")
            #     trade.close()


# buy sell signals
def is_long_close_signal(self):
    return True if self.data.total_signal[-1] == 1 else False
    # return True if self.data['ema200_signal'] == 1 else False
    # return True if self.data.RSI[-1]>=75 else False


# def is_short_close_signal(self):
#     return True if self.data['ema200_signal'] == 1 else False


def is_long_buy_signal(self):
    return True if self.data.total_signal[-1] == 2 else False


# def is_short_sell_signal(self):
#     return True if self.data.total_signal[-1] != 0 and self.data['ema200_signal'] == 1 else False
class Plotlines:
    pass

class Teststrategy(Strategy):
    data_g = []
    mysize = []
    # buy_dollars = 1000
    # buy_period = 100
    count = 0
    # cash_total = 0

    def init(self):
        """ create plotlines based on presets """
        super().init()
        self.plotlines = Plotlines()

        _LOG.debug(f"initiating backtest")
        for idx, column_name in enumerate(self.data_g['s_ind'].get('indicators')):
            setattr(self.plotlines, f'{column_name}',
                    self.I(get_data, self.data, column_name,
                           color=self.data_g['s_ind'].get('colors')[idx],
                           scatter=self.data_g['s_ind'].get('scatter')[idx],
                           overlay=self.data_g['s_ind'].get('overlay')[idx]))
        self.mysize = self.data_g['initsize']
        self.days_max_order_to_fulfill = self.data_g['days_max_order_to_fulfill']

    def next(self):
        super().next()
        self.count += 1
        today_idx = self.data._Data__i - 1
        _LOG.debug(f"current candle {today_idx}")

        # determine if close trades

        limit_order_duration_on_exchange(self, self.days_max_order_to_fulfill, today_idx)
        create_close_on_trades(self)

        if is_long_buy_signal(self):
            create_buy_order(self)

        # elif is_short_sell_signal(self) and len(self.trades) == 0:
        #     create_sell_order(self)

if __name__ == "__main__":
    df, data_g, indicators0, colors0, \
    scatter0, sub_indicators0, sub_colors0, \
    colors, overlay, indicators, scatter, sub2_ind0, sub2_col0, \
    init_data_g = get_signals()

    # initiate preplot
    dfpl_plot = df.copy()
    # dfpl_plot = df[1000:1200].copy()
    from doc.examples.utilities import plot_visualization

    metric_figure, fig, bottom_figure = \
        plot_visualization(dfpl_plot, indicators0,
                           colors0, scatter0, sub_indicators0, sub_colors0, sub2_ind0, sub2_col0)

    metric_figure.show()
