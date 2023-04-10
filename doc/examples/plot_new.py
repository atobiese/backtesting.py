#!/usr/bin/env python3

import mplfinance as mpf
mpf.__version__
import numpy as np
import pandas as pd
import yfinance as yf

from strategies.strategy_dca import get_signals
# btc = yf.download('BTC-USD', '2014-09-01')
df, data_g, indicators0, colors0, \
scatter0, sub_indicators0, sub_colors0, \
colors, overlay, indicators, scatter, sub2_ind0, sub2_col0, \
init_data_g = get_signals()
print('''

Finally, as a more pratical example, we use `fill_between` to color a MACD plot:

''')

df = df

# =======
#  MACD:

exp12     = df['Close'].ewm(span=12, adjust=False).mean()
exp26     = df['Close'].ewm(span=26, adjust=False).mean()
exp200    = df['Close'].ewm(span=200, adjust=False).mean()
macd      = exp12 - exp26
signal    = macd.ewm(span=9, adjust=False).mean()
histogram = macd - signal

fb_12up = dict(y1=exp12.values,y2=exp26.values,where=exp12>exp26,color="#93c47d",alpha=0.6,interpolate=True)
fb_12dn = dict(y1=exp12.values,y2=exp26.values,where=exp12<exp26,color="#e06666",alpha=0.6,interpolate=True)
fb_exp12 = [fb_12up,fb_12dn]

fb_macd_up = dict(y1=macd.values,y2=signal.values,where=signal<macd,color="#93c47d",alpha=0.6,interpolate=True)
fb_macd_dn = dict(y1=macd.values,y2=signal.values,where=signal>macd,color="#e06666",alpha=0.6,interpolate=True)
fb_macd_up['panel'] = 1
fb_macd_dn['panel'] = 1

fb_macd = [fb_macd_up,fb_macd_dn]

apds = [mpf.make_addplot(exp12,color='lime'),
        mpf.make_addplot(exp26,color='c'),
        mpf.make_addplot(exp200,color='c'),
        mpf.make_addplot(histogram,type='bar',width=0.7,panel=1,
                         color='dimgray',alpha=0.65,secondary_y=True),
        mpf.make_addplot(macd,panel=1,color='fuchsia',secondary_y=False),
        mpf.make_addplot(signal,panel=1,color='b',secondary_y=False)#,fill_between=fb_macd),
       ]

s = mpf.make_mpf_style(base_mpf_style='blueskies',facecolor='aliceblue',rc={'figure.facecolor':'lightgrey'})

mpf.plot(df,type='candle',addplot=apds,figscale=1.6,figratio=(1,1),title='\n\nMACD',
         style=s,volume=True,volume_panel=2,panel_ratios=(3,4,1),tight_layout=True,
         fill_between=fb_macd+fb_exp12)