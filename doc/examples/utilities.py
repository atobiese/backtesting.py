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
# start logging
logging.debug("Begin")
_LOG.setLevel(logging.DEBUG)


def create_dir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("created directory %s: ", directory)


def fetch_download_data(start, end, symbol, load_from_csv, csv_directory):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    if load_from_csv:
        data = pd.read_csv(os.path.join(csv_directory,
                                        f'{symbol}.csv'), index_col=['Date'], parse_dates=True)
    else:
        data = yf.download(symbol, start, end)
        save_file = True
        if save_file:
            create_dir_if_not_exist(csv_directory)
            cl_lower = symbol
            abs_filename = os.path.join(csv_directory, cl_lower)
            data.to_csv(f'{abs_filename}.csv')
    return data


# convert to weekly candles
def convert_to_weekly(df):
    from datetime import timedelta
    df = df.reset_index()
    df["day_of_week"] = df["Date"].dt.weekday
    df["1st_day_of_week"] = df["Date"] - df["day_of_week"] * timedelta(days=1)
    df_weekly = df.copy(deep=True)
    # df["Date"] = pd.to_datetime(df["Date"])
    df_weekly.drop(columns=["Date"], inplace=True)
    df_weekly.rename(columns={"1st_day_of_week": "Date"}, inplace=True)
    df_weekly = df_weekly[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    ].groupby(["Date"]).agg(
        {"Open": "first", "High": ["max"], "Low": ["min"], "Close": "last", "Adj Close": "last", "Volume": ["sum"],
         })
    df_weekly.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    # copy back
    df = df_weekly.copy(deep=True)
    return df

def limit_order_duration_on_exchange(self, days_max_order_to_fulfill, today_idx):
    for j, order in enumerate(self.orders):
        # we ignore the order if it is fulfilled and a trade is created
        # is contingent means it has been fulfilled
        if order.is_contingent:
            # ignore the fulfilled orders from the list, only non-fullfilled orders will be canceelled
            ordertype ="long" if order.parent_trade.is_long is True else 'short'
            _LOG.debug(
                f'Order tag candle nr: {order.tag} '
                f'fulfilled and created trade: {ordertype} limit order with size {order.parent_trade.size} '
                f'after {order.parent_trade.entry_bar - order.tag} candles, '
                f'entry price {order.parent_trade.entry_price} '
                f'open {self.data.Open[order.parent_trade.entry_bar]} '
                f'low price: {self.data.Low[order.parent_trade.entry_bar]}')
            pass
        # if its not fulfilled, check how long it has been on the exchange
        else:
            # allow the order to stay on exchange for given days before its cancelled
            if today_idx - order.tag >= days_max_order_to_fulfill:
                _LOG.debug(
                    f'non-fulfilled limit order at {order.limit} expires now: {order.tag} after {today_idx - order.tag} days')
                order.cancel()
                # self.ordertime_idx.pop(j)
                _LOG.debug(f"total orders: {len(self.orders)}")
                break


def get_data(data, column_name):
    if not hasattr(data, column_name):
        _LOG.error(f'indicator does not exist in dataframe {column_name}')
        # return as a df series
    return data.df[column_name]

def get_data2(data, column_names):
    for name in column_names:
        if not hasattr(data, name):
            _LOG.error(f'indicator does not exist in dataframe {name}')
    #     # return as a df series
    return data.df[column_names[0]], data.df[column_names[1]]

# def get_data(data, column_name):
#     if not column_name in data.df[0].columns :
#         _LOG.error(f'indicator does not exist in dataframe {column_name}')
#         # return as a df series
#     return data.df[0][column_name]
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pio.renderers.default = 'browser'
def plot_visualization(df, indicators, colors, scatter, sub_indicators, sub_colors, sub2_ind0, sub2_col0):
    # Data Visualization
    # main_plots = [go.Candlestick(x=dfpl.index,
    #                              open=dfpl['Open'],
    #                              high=dfpl['High'],
    #                              low=dfpl['Low'],
    #                              close=dfpl['Close'],), ]
    df = df.reset_index()
    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        #subplot_titles=(f'{ticker}', '', 'MACD'),
        row_width=[0.2, 0.3, 0.2, 0.3, 0.2, 0.7]
    )
    a=1
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            showlegend=False
            #name='Candlestick chart'
        ),
        row=1,
        col=1,
    )
    green = '#3D9970'
    red = '#FF4136'
    # fig = go.Figure(data=main_plots)
    for idx, _indicator in enumerate(indicators):
            if not scatter[idx]:
                # fig.add_scatter(x=dfpl.index, y=dfpl[_indicator],
                #                line=dict(width=1, color=colors[idx]), name=_indicator)
                fig.add_trace(
                    go.Scatter(x=df['Date'], y=df[_indicator], name=_indicator,
                               line=dict(color=colors[idx], width=1)),
                    row=1,
                    col=1,
                )
                continue
            if scatter[idx]:
                # fig.add_scatter(x=dfpl.index, y=dfpl[_indicator], mode="markers",
                #                 marker=dict(size=10, color=colors[idx]), name=_indicator)
                fig.add_trace(
                    go.Scatter(x=df['Date'], y=df[_indicator], name=_indicator,  mode='markers'),
                    row=1,
                    col=1,
                )
                continue

            # Plot volume trace on 2nd row
    colors = [red if row['Open'] - row['Close'] >= 0
              else green for index, row in df.iterrows()]

    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color=colors),
        row=2,
        col=1,
    )
    # Plot MACD trace on 3rd row
    colors = [green if val >= 0
              else red for val in df['MACDh_12_26_9']]
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['MACD_12_26_9'], name='MACD_12_26_9',
                   line=dict(color=red, width=1)),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['MACDs_12_26_9'], name='MACDs_12_26_9',
                   line=dict(color='Black', width=1)),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=df['Date'], y=df['MACDh_12_26_9'], name='MACDh_12_26_9',
               marker_color=colors),
        row=3,
        col=1,
    )

    for idx, _indicator in enumerate(sub2_ind0):
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df[_indicator], name=_indicator,
                       line=dict(color=sub2_col0[idx], width=1)),
            row=4,
            col=1,
        )

    colors = [green if row['RSI'] - row['RSI_upper_band'] >= 0
              else red for index, row in df.iterrows()]

    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['RSI_lower_band'], name='RSI_lower_band',
                   marker_color=colors),
        row=5,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['RSI_upper_band'], name='RSI_upper_band', marker_color=colors),
        row=5,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', marker_color=colors),
        row=5,
        col=1,
    )
    for idx, _indicator in enumerate(sub_indicators):
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df[_indicator], name=_indicator,
                       line=dict(color=sub_colors[idx], width=1)),
            row=6,
            col=1,
        )
    # removing all empty dates
    # build complete timeline from start date to end date
    dt_all = pd.date_range(start=df.index[0], end=df.index[-1])
    # retrieve the dates that ARE in the original datset
    dt_obs = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(df.index)]
    # define dates with missing values
    dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
    # fig.update_layout(xaxis_rangebreaks=[dict(values=dt_breaks)])

    # fig['layout']['xaxis2']['title'] = 'Date'
    # fig['layout']['yaxis']['title'] = 'Price'
    # fig['layout']['yaxis2']['title'] = 'Volume'
    # fig['layout']['yaxis3']['title'] = 'MACD'

    fig.update_xaxes(
        # rangebreaks=[{'bounds': ['sat', 'mon']}],
        rangeslider_visible=False,
    )

    # removing white spaces
    fig.update_layout(margin=go.layout.Margin(
        l=20,  # left margin
        r=20,  # right margin
        b=20,  # bottom margin
        t=20  # top margin
    ))
    # update layout by changing the plot size, hiding legends & rangeslider, and removing gaps between dates
    fig.update_layout(height=900, width=1800,
                      showlegend=True,
                      xaxis_rangeslider_visible=False,
                      xaxis_rangebreaks=[dict(values=dt_breaks)])

    # update y-axis label
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="MACD", showgrid=False, row=3, col=1)
    fig.update_yaxes(title_text="Signal", row=4, col=1)
    fig.update_yaxes(title_text="RSI", row=5, col=1)
    fig.update_yaxes(title_text="KST", row=6, col=1)
    return fig


def update_plots_with_signals(_trades, fig):
    # fig.add_scatter(x=_trades['EntryTime'], y=_trades.EntryPrice, mode="markers",
    #                 marker=dict(size=10, color="green"), name="Entrybar")
    fig.add_trace(
        go.Scatter(x=_trades['EntryTime'], y=_trades.EntryPrice, mode='markers',
                   marker=dict(size=10, color="green"), name="Entrybar"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=_trades['ExitTime'], y=_trades.ExitPrice, mode='markers',
                   marker=dict(size=10, color="Red"), name="Exitbar"),
        row=1,
        col=1,
    )

    # fig.add_scatter(x=_trades['ExitTime'], y=_trades.ExitPrice, mode="markers",
    #                 marker=dict(size=10, color="Red"), name="Exitbar")
    # # for t in fig.data[0:2]:
    #     metric_figure.append_trace(t, row=1, col=1)
    # metric_figure.show()
    return fig

def send_email(*args ):
    import pandas as pd
    from email.message import EmailMessage
    import smtplib

    email_from = "atobiese@yahoo.no"
    password = "bcclguzwwfwozyke"
    email_to = "atobiese@yahoo.no"
    # email_to = ["atobiese@yahoo.no", "andrew.tobiesen@sintef.no"]

    url= "C:/repos/backtestingpy/Teststrategy_data_g-_.html"

    # url_file = "file:///C:/repos/backtestingpy/Teststrategy_data_g.html"
    # fp = urllib.request.urlopen(url_file)
    # mybytes = fp.read()
    # mystr = mybytes.decode("utf8")
    # fp = open(url)
    # mystr = fp.read()
    # fp.close()

    date_str = pd.Timestamp.today().strftime('%Y-%m-%d')
    email = EmailMessage()
    email["From"] = email_from
    email["To"] = email_to
    email['Subject'] = f'Strategy Report: - {date_str}'

    report = ""
    for parts in args:
        report += parts
        report += "<br><br>"

    email.set_content(report, subtype="html")

    with open(url, "rb") as f:
        email.add_attachment(
            f.read(),
            filename="technicals.html",
            maintype="text",
            subtype="html"
        )

    smtp = smtplib.SMTP_SSL('smtp.mail.yahoo.com', 465)
    smtp.login(email_from, password)
    smtp.sendmail(email_from, email_to, email.as_string())
    smtp.quit()

def get_yahoo_data(stck='SPY', load_from_csv=False):
    from dateutil.relativedelta import relativedelta
    # from fin import fetch_download_data
    from datetime import datetime, date
    # load_from_csv = False
    # # stck = '^GSPC'
    # stck = 'SPY'
    # # stck = 'EQNR.OL'
    # # stck = '^RUI'
    # # stck = 'EURUSD=X'
    # # stck = 'TROW'
    now = datetime.now()
    # now_minus_10 = now + relativedelta(years=-4)
    start = datetime.now() + relativedelta(years=-3)
    end = now  # datetime.now() + relativedelta(years=-2)

    # start = date(2002, 1, 1)
    end = now #date(2023, 3, 6 + 1)
    # end = date.today()
    # # get ticker data for the last 10  years
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Project root
    CSV_DIR = os.path.join(ROOT_DIR, "../doc/examples/temp_csv")
    df = fetch_download_data(start, end, stck, load_from_csv=load_from_csv, csv_directory=CSV_DIR)
    return df