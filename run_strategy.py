import logging

from backtesting import Backtest
from backtesting import Strategy
from doc.examples.utilities import limit_order_duration_on_exchange, get_data, update_plots_with_signals
from apscheduler.schedulers.blocking import BlockingScheduler
_LOG = logging.getLogger(__name__)
# run this to start logging
logging.debug("Begin")
_LOG.setLevel(logging.DEBUG)


def _import(path, package):
    # import importlib
    # """Import the given plugins file from package"""
    # return importlib.import_module(package, package=package
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(package, path)
    foo = importlib.util.module_from_spec(spec)
    # sys.modules[package] = foo
    spec.loader.exec_module(foo)
    return foo


def run_backtest(strategy_name):
    """ main script to run strategy based on file strategy name as input"""
    from definitions import STRAT_DIR
    import os

    strat = os.path.join(STRAT_DIR, f'{strategy_name}.py')
    mod = _import(strat, strategy_name)

    df, data_g, indicators0, colors0, \
    scatter0, sub_indicators0, sub_colors0, \
    colors, overlay, indicators, scatter, sub2_ind0, sub2_col0, \
    init_data_g = mod.get_signals()

    # activate functions used for strategy
    create_buy_order = mod.create_buy_order
    # create_sell_order = mod.create_sell_order
    create_close_on_trades = mod.create_close_on_trades
    # is_long_close_signal = mod.is_long_close_signal
    # is_short_close_signal = mod.is_short_close_signal
    is_long_buy_signal = mod.is_long_buy_signal
    # is_short_sell_signal = mod.is_short_sell_signal

    # initiate preplot
    dfpl = df.copy()
    # dfpl_plot = df[1000:1200].copy()
    dfpl_plot = df.copy()
    from doc.examples.utilities import  plot_visualization
    fig = \
        plot_visualization(dfpl_plot, indicators0,
                           colors0, scatter0, sub_indicators0, sub_colors0,
                           sub2_ind0, sub2_col0)

    # metric_figure.show()

    s_ind = {
        "colors": colors,
        "overlay": overlay,
        "indicators": indicators,
        "scatter": scatter
    }
    data_g['s_ind'] = s_ind

    class Plotlines:
        pass

    class Teststrategy(Strategy):
        data_g = []
        mysize = []
        buy_dollars = 1000
        buy_period = 100
        count = 0
        cash_total = 0

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

            # limit_order_duration_on_exchange(self, self.days_max_order_to_fulfill, today_idx)
            create_close_on_trades(self)

            if is_long_buy_signal(self):
                create_buy_order(self)

            # elif is_short_sell_signal(self) and len(self.trades) == 0:
            #     create_sell_order(self)

    def _run_backtest(dfpl, init_data_g, fig,
                     data_g, dfpl_plot):
        # run the backtesting engine
        bt = Backtest(dfpl, mod.Teststrategy,
                      exclusive_orders=init_data_g['exclusive_orders'],
                      cash=init_data_g['cash'], margin=init_data_g['margin'],
                      commission=init_data_g['commission'],
                      trade_on_close = init_data_g['trade_on_close'])

        stat = bt.run(data_g=data_g)
        print(stat)

        print(stat._trades)
        print(stat._strategy)
        btfig = bt.plot(plot_width=1500, smooth_equity=True,
                superimpose=True, reverse_indicators=False,
                open_browser=True, plot_drawdown=True)

        # stat_opt = bt.optimize(n1=range(5, 30, 5),
        #                     n2=range(10, 70, 5),
        #                     maximize='Equity Final [$]',
        #                     constraint=lambda param: param.n1 < param.n2)

        # stat_opt = bt.optimize(days_max_order_to_fulfill=range(5, 40, 5),
        #                     maximize='Equity Final [$]')

        # update pre_plot with result signals from backtest
        date_strt, date_end = dfpl_plot.index[0], dfpl_plot.index[-1]
        _trades_entry = []
        for idx, ele in enumerate(stat._trades['EntryTime']):
            # checking for date in range
            if ele >= date_strt and ele <= date_end:
                _trades_entry.append(idx)

        stat_cp = stat.copy()
        a_trades = stat_cp._trades.iloc[_trades_entry]
        fig = update_plots_with_signals(a_trades, fig)
        fig.show()

        import pandas as pd
        # make some more todays data
        last_day = dfpl.iloc[-1]

        total_signal = last_day.loc['total_signal']
        if total_signal == 0:
            trigger = f"strategy suggests no changes, signal == {total_signal}"
        elif total_signal == 1:
            trigger = f"strategy suggests saying out of market, signal == {total_signal}"
        elif total_signal == 2:
            trigger = f"strategy suggests going long, signal == {total_signal}"

        day = last_day.name
        date_today = day.strftime("%Y-%m-%d %H:%M:%S")
        last_day.loc["todays date"] = date_today
        last_day.loc["current trigger"] = trigger
        df_last_day = pd.DataFrame(last_day)
        df_last_day.to_html()

        sendemail = False
        if sendemail:
            import pandas as pd
            df = pd.DataFrame(stat[0:28], columns=['value'])
            df_trades = pd.DataFrame(stat['_trades'])
            from doc.examples.utilities import send_email
            send_email(df.to_html(), df_trades.to_html(), init_data_g['strategy_text'], df_last_day.to_html())

        return btfig

    _run_backtest(dfpl, init_data_g, fig, data_g, dfpl_plot)


def backtest_job(strategy_name):

    scheduler = BlockingScheduler()
    # scheduler.add_job(run_backtest, args=[strategy_name], trigger = 'cron', day_of_week='mon-fri', hour='00-23', minute='43,44,45,46, 47, 48', start_date='2023-03-21 12:00:00', timezone='America/Chicago')
    # scheduler.add_job(run_backtest, args=[strategy_name], trigger = 'cron', day_of_week='mon-fri', hour='00-23', minute='43,44,45,46, 47, 48', start_date='2023-03-21 12:00:00', timezone='America/Chicago')
    # scheduler.add_job(run_backtest, args=[strategy_name], trigger='cron', day_of_week='mon-fri', hour='00-23',
    #                   minute='*', start_date='2023-03-21 12:00:00', timezone='America/Chicago')
    scheduler.add_job(run_backtest, args=[strategy_name], trigger='cron', hour='12-23/2', timezone='Europe/Oslo')

    scheduler.start()


if __name__ == "__main__":
    # strategy_name = 'teststrategy'
    # strategy_name = 'strategy_macd'
    # strategy_name = 'strategy_dca'
    # strategy_name = 'strategy_sw'
    # strategy_name = 'strategy_monthly'
    # strategy_name = 'over200sma_trow'
    # from doc.examples.utilities import send_email
    # send_email(None)
    # strategy_name = 'over200sma_spy'
    # strategy_name = 'over200sma'
    strategy_name = 'weekly_over20_10_ema_spy'

    run_backtest(strategy_name)
    # backtest_job(strategy_name)
