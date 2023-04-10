
# import pandas_datareader.data as web
import datetime
from doc.examples.utilities import  plot_visualization, update_plots_with_signals
from doc.examples.utilities import  limit_order_duration_on_exchange, get_data, fetch_download_data
import os
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
    end = datetime.date(2021, 4, 16 + 1)

    # # get ticker data for the last 10  years
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Project root
    CSV_DIR = os.path.join(ROOT_DIR, "../doc/examples/temp_csv")
    data = fetch_download_data(start, end, stck, load_from_csv=load_from_csv, csv_directory=CSV_DIR)
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

from backtesting import Backtest, Strategy # バックテスト、ストラテジー
from backtesting.lib import crossover

import pandas_ta as ta

def MACD(close, n1, n2, ns):
    macd = ta.macd(close.df['Close'], n1, n2, ns)
    macd1 = macd['MACD_12_26_9']
    macdsignal = macd['MACDs_12_26_9']

    return macd1, macdsignal

class MACDCross(Strategy):
    n1 = 12 #短期EMAの期間
    n2 = 26 #長期EMAの期間
    ns = 9 #シグナル（MACDのSMA）の期間

    def init(self):
        self.macd, self.macdsignal = self.I(MACD, self.data.Close, self.n1, self.n2, self.ns)

    def next(self): # チャートデータの行ごとに呼び出される
        if crossover(self.macd, self.macdsignal): #macdがsignalを上回った時
            self.buy() # 買い
        elif crossover(self.macdsignal, self.macd): #signalがmacdを上回った時
            self.position.close() # 売り

# バックテストを設定
bt = Backtest(
    data, # チャートデータ
    MACDCross, # 売買戦略
    cash=1000, # 最初の所持金
    commission=0.00495, # 取引手数料
    margin=1.0, # レバレッジ倍率の逆数（0.5で2倍レバレッジ）
    trade_on_close=True, # True：現在の終値で取引，False：次の時間の始値で取引
    exclusive_orders=True #自動でポジションをクローズ(オープン)
)

output = bt.run() # バックテスト実行
print(output) # 実行結果(データ)
bt.plot(plot_width=1500, smooth_equity=True,
        superimpose=True, reverse_indicators=False,
        open_browser=True, plot_drawdown=False)