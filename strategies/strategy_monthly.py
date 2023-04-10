from backtesting import Strategy
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta as rd
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
from definitions import STRAT_DIR

strat = os.path.join(STRAT_DIR, 'dataframe_test.csv')


def get_signals():
    _load_from_csv = False
    if _load_from_csv:
       df = pd.read_csv(os.path.join(strat), index_col=['Date'], parse_dates=True)
    else:

        dummy_data = False
        if not dummy_data:
            load_from_csv = False
            stck = '^GSPC'
            # stck = '^RUI'
            # stck = 'EURUSD=X'
            # stck = 'AAPL'
            # now = datetime.now()
            start = dt.datetime.now() + rd(years=-15)
            end = dt.date.today()

            # start = dt.datetime.date(2018, 1, 2)
            # end = dt.datetime.date(2021, 4, 16 + 1)

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
        df['RSI'] = ta.rsi(df.Close, length=12)
        df["RSI_lower_band"] = df['Close'] * 0 + 30.0
        df["RSI_upper_band"] = df['Close'] * 0 + 70.0

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
        #     return

        import time
        import datetime
        from pandas.tseries.holiday import USFederalHolidayCalendar
        from pandas.tseries.offsets import CustomBusinessDay
        from dateutil.relativedelta import relativedelta

        # strategy
        # https://therobusttrader.com/trading-the-end-of-the-month-strategies-that-works-for-traders/
        #     The rule of the strategy
        # The rule of the strategy is to go long at the close on the fifth last trading day of
        # the month, and we exit after seven days (at the close of the third trading day of the next month).
        # Exposure time is around 33%, but it performs very well in most markets and frequently beats the buy and hold
        # strategy despite its low exposure time.

        def get_strategy_dates(_df=None, start__month_day=3, end_month_day=5):
            # Create dates needed to be entered as parameters
            if _df is None:
                today = datetime.date.today()
            else:
                today = _df
            first = today.replace(day=1)
            # End of the Prior Month
            # eopm = first - datetime.timedelta(days=1)
            # eopm = eopm.strftime("%Y%m%d")
            # Create first business day of current month date
            us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
            focm = first
            nxtMo = today + relativedelta(months=+1)
            fonm = nxtMo.replace(day=1)
            eocm = fonm - datetime.timedelta(days=1)
            montly_bd = pd.date_range(start=focm, end=eocm, freq=us_bd)
            montly_bd_str = montly_bd.strftime("%Y%m%d")
            # First Business Day of the Month
            first_bd = montly_bd_str[0]
            # Last Business Day of the Month
            third_montly_bd = montly_bd[3 - 1]
            lst_fifth_day = montly_bd[-5]
            # last_bd = first_bd[lst_fifth_day]

            return third_montly_bd, lst_fifth_day

        third_montly_bd, lst_fifth_day = get_strategy_dates(None, 3, 5)
        a = 1

        def _total_signal_next(l):
            date = df.index[l]
            third_montly_bd, lst_fifth_day = get_strategy_dates(date, 3, 5)
            a = 1
            if (
                    # crossover(df['MACD_12_26_9'][l - 2:l], df['MACDs_12_26_9'][l - 2:l])
                    date == lst_fifth_day
                    # df['MACDs_12_26_9'][l] < df['MACD_12_26_9'][l]
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
                    date == third_montly_bd
                    # df.ema200_signal[l] == 1
                    # and df.ema20_signal[l] == 1
                    # and df.Close[l] >= df['BBU_14_2.0'][l]
                    # df['MACDs_12_26_9'][l] > df['MACD_12_26_9'][l]
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
                return x['Low'] - x['Low'] * 0.01

        def pointnegbreak(x):
            if x['total_signal'] == 1:
                return x['High'] + x['Low'] * 0.01

        df['pointposbreak'] = df.apply(lambda row: pointposbreak(row), axis=1)
        df['pointnegbreak'] = df.apply(lambda row: pointnegbreak(row), axis=1)
        #save
        df.to_csv(strat)


    data_g = {
        "initsize": 0.9999999999,
        "TPSLRatio": 1.5,
        "perc": 0.02,
        "days_max_order_to_fulfill": 5
    }

    # When an order is executed or [filled], it results in a `Trade`.
    #
    # If you wish to modify aspects of a placed but not yet filled order,
    # cancel it and place a new one instead.
    # All placed orders are [Good 'Til Canceled].

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
        "cash": 100_000,
        "margin": 1.0,
        "commission": 0.00,
        "exclusive_orders": False,
        "trade_on_close": True,
    }


    return df, data_g, indicators0, colors0, \
           scatter0, sub_indicators0, sub_colors0, \
           colors, overlay, indicators, scatter, sub2_ind0, sub2_col0, \
           init_data_g


def create_buy_order(self):
    if len(self.trades) == 0:
        # self.buy()
        # Cancel previous orders
        # for order in self.orders:
        #     order.cancel()
        # sl1 = price_close - slatr
        # tp1 = price_close + slatr * TPSLRatio
        TPSLRatio = self.data_g['TPSLRatio']
        perc = self.data_g['perc']

        today_idx = self.data._Data__i - 1
        today = self.data.index[-1]
        _LOG.debug(f"current candle {today_idx}")
        price_close = self.data.Close[-1]


        percent = 0.0
        _price_limit = self.data.Close[-1] - (self.data.Close[-1] * percent)

        # vi ønsker å kjøpe uten limit, der ordre eksekveres dagen etter
        price_limit = None
        sl_limit = _price_limit
        # Cancel previous order(s) as we only want one order at a time
        # decied to close previous and open a new order
        keep_previous_open_orders = False
        if not keep_previous_open_orders:
            for order in self.orders:
                order.cancel()
            # sl1 = min(self.data.Low[-1], self.data.Low[-2]) * (1 - perc)
            # tp1 = price_close + (price_close - sl1) * TPSLRatio
            # self.buy(sl=tp1/2, tp=tp1, size=self.mysize, tag=today_idx + 1)
            _LOG.debug(f"buyorder at bar: {today_idx}, limitprice: {price_limit}, close price: {price_close}")
            self.buy(sl=sl_limit / 2, limit=price_limit, size=self.mysize, tag=today_idx)
            # self.buy()
        else:
            if len(self.orders) > 0:
                return
            else:
                _LOG.debug(f"buyorder at bar: {today_idx}, limitprice: {price_limit}, close price: {price_close}")
                self.buy(sl=sl_limit / 2, limit=price_limit, size=self.mysize, tag=today_idx)


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
    # dfpl_plot = df.copy()
    dfpl_plot = df[500:1200].copy()
    from doc.examples.utilities import plot_visualization

    fig = \
        plot_visualization(dfpl_plot, indicators0,
                           colors0, scatter0, sub_indicators0, sub_colors0, sub2_ind0, sub2_col0)

    fig.show()
