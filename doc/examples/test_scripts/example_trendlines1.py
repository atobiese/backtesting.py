import numpy as np
import pandas as pd
import yfinance as yf


def trend_filter(df: pd.DataFrame,
                 growth_4_min: float = 25.,
                 growth_12_min: float = 50.,
                 growth_24_min: float = 80.) -> np.array:
    '''
    Take in a pandas series and output a binary array to indicate if a stock
    fits the growth criteria (1) or not (0)0

    Parameters
    ----------
    prices : pd.core.series.Series
        The prices we are using to check for growth
    growth_4_min : float, optional
        The minimum 4 week growth. The default is 25
    growth_12_min : float, optional
        The minimum 12 week growth. The default is 50
    growth_24_min : float, optional
        The minimum 24 week growth. The default is 80

    Returns
    -------
    np.array
        A binary array showing the positions where the growth criteria is met

    '''

    growth_func = lambda x: 100 * (x.values[-1] / x.min() - 1)

    growth_4 = df.rolling(20).apply(growth_func) > growth_4_min
    growth_12 = df.rolling(60).apply(growth_func) > growth_12_min
    growth_24 = df.rolling(120).apply(growth_func) > growth_24_min

    return np.where(
        growth_4 | growth_12 | growth_24,
        1,
        0,
    )


# if __name__ == '__main__':
#     df = yf.download('TSLA').reset_index()
#     df.loc[:, 'trend_filter'] = trend_filter(df)
#     df.dropna()

import numpy as np
import numba as nb


@nb.jit(nopython=True)
def explicit_heat_smooth(prices: np.array,
                         t_end: float = 5.0) -> np.array:
    '''
    Smoothen out a time series using a explicit finite difference method.

    Parameters
    ----------
    prices : np.array
        The price to smoothen
    t_end : float
        The time at which to terminate the smootheing (i.e. t = 2)

    Returns
    -------
    P : np.array
        The smoothened time-series
    '''

    k = 0.1  # Time spacing, must be < 1 for numerical stability

    # Set up the initial condition
    P = prices

    t = 0
    while t < t_end:
        # Solve the finite difference scheme for the next time-step
        P = k * (P[2:] + P[:-2]) + P[1:-1] * (1 - 2 * k)

        # Add the fixed boundary conditions since the above solves the interior
        # points only
        P = np.hstack((
            np.array([prices[0]]),
            P,
            np.array([prices[-1]]),
        ))
        t += k

    return P


@nb.jit(nopython=True)
def check_consolidation(prices: np.array,
                        perc_change_days: int,
                        perc_change_thresh: float,
                        check_days: int) -> int:
    '''
    Smoothen the time-series and check for consolidation, see the
    docstring of find_consolidation for the parameters
    '''

    # Find the smoothed representation of the time series
    prices = explicit_heat_smooth(prices)

    # Perc change of the smoothed time series to perc_change_days days prior
    perc_change = prices[perc_change_days:] / prices[:-perc_change_days] - 1

    consolidating = np.where(np.abs(perc_change) < perc_change_thresh, 1, 0)

    # Provided one entry in the last n days passes the consolidation check,
    # we say that the financial instrument is in consolidation on the end day
    if np.sum(consolidating[-check_days:]) > 0:
        return 1
    else:
        return 0


@nb.jit(nopython=True)
def find_consolidation(prices: np.array,
                       days_to_smooth: int = 50,
                       perc_change_days: int = 5,
                       perc_change_thresh: float = 0.015,
                       check_days: int = 5) -> np.array:
    '''
    Return a binary array to indicate whether each of the data-points are
    classed as consolidating or not

    Parameters
    ----------
    prices : np.array
        The price time series to check for consolidation
    days_to_smooth : int, optional
        The length of the time-series to smoothen (days). The default is 50.
    perc_change_days : int, optional
        The days back to % change compare against (days). The default is 5.
    perc_change_thresh : float, optional
        The range trading % criteria for consolidation. The default is 0.015.
    check_days : int, optional
        This says the number of lookback days to check for any consolidation.
        If any days in check_days back is consolidating, then the last data
        point is said to be consolidating. The default is 5.

    Returns
    -------
    res : np.array
        The binary array indicating consolidation (1) or not (0)

    '''

    res = np.full(prices.shape, np.nan)

    for idx in range(days_to_smooth, prices.shape[0]):
        res[idx] = check_consolidation(
            prices=prices[idx - days_to_smooth:idx],
            perc_change_days=perc_change_days,
            perc_change_thresh=perc_change_thresh,
            check_days=check_days,
        )

    return res


if __name__ == '__main__':
    # df = yf.download('TSLA').reset_index()
    # df.loc[:, 'consolidating'] = find_consolidation(df['Close'].values)
    # df.dropna()
    from doc.examples.utilities import get_yahoo_data, convert_to_weekly
    # df = yf.download('TSLA').reset_index()
    load_from_csv = True
    # stck = '^GSPC'
    # stck = 'SPY'
    # stck = 'EQNR.OL'
    # stck = '^RUI'
    # stck = 'EURUSD=X'
    stck = 'STM'
    df = get_yahoo_data(stck, load_from_csv)
    # df = df.reset_index()
    df = convert_to_weekly(df)
    df = df.reset_index()

    from dateutil.relativedelta import relativedelta
    from datetime import datetime, date
    now = datetime.now()
    start = datetime.now() + relativedelta(years=-2)
    end = now

# '''
#     Parameters
#     ----------
#     prices : np.array
#         The price time series to check for consolidation
#     days_to_smooth : int, optional
#         The length of the time-series to smoothen (days). The default is 50.
#     perc_change_days : int, optional
#         The days back to % change compare against (days). The default is 5.
#     perc_change_thresh : float, optional
#         The range trading % criteria for consolidation. The default is 0.015.
#     check_days : int, optional
#         This says the number of lookback days to check for any consolidation.
#         If any days in check_days back is consolidating, then the last data
#         point is said to be consolidating. The default is 5.
# '''

    df.loc[:, 'consolidating'] = find_consolidation(df['Close'].values,
                                                   days_to_smooth= 50,
                                                   perc_change_days= 1,
                                                       perc_change_thresh= 0.1,
                                                   check_days = 10 )


    df.loc[:, 'trend_filter'] = trend_filter(df['Close'])
    df.loc[:, 'filtered'] = np.where(
        df['consolidating'] + df['trend_filter'] == 2,
        True,
        False,
    )

    import plotly.io as pio
    import plotly.graph_objects as go

    pio.renderers.default = 'browser'


    df = df[(df['Date'] > start) & (df['Date'] <= end)]

    df['bar_plot'] = np.where(
        df['filtered'],
        df['High'].max(),
        df['Low'].min(),
    )

    fig = go.Figure()

    df['bar_plot_cons'] = np.where(
        df['consolidating'],
        df['High'].max(),
        df['Low'].min(),
    )

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            name=stck,
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            showlegend=True,
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['bar_plot'],
            fill='tozeroy',
            name="consolidatioin and trend filter",
            fillcolor='rgba(0, 236, 109, 0.2)',
            mode='markers',
            # line={'width': 0, 'shape': 'hvh'},
            showlegend=True,
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['bar_plot_cons'],
            fill='tozeroy',
            name="consolidatioin",
            fillcolor='rgba(0, 136, 109, 0.2)',
            mode='lines',
            line={'width': 0, 'shape': 'hvh'},
            showlegend=True,
        ),
    )

    fig.update_layout(
        xaxis={'title': 'Date'},
        yaxis={'range': [df['Low'].min(), df['High'].max()], 'title': 'Price'},
        title='Scanned Breakout Candidates'
    )

    fig.update_xaxes(
        rangebreaks=[{'bounds': ['sat', 'mon']}],
        rangeslider_visible=False,
    )

    fig.show()