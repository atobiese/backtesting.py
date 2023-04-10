import logging

from backtesting import Backtest
from backtesting import Strategy
from utilities import limit_order_duration_on_exchange, get_data, update_plots_with_signals

_LOG = logging.getLogger(__name__)
# run this to start logging
logging.debug("Begin")
_LOG.setLevel(logging.DEBUG)


def _import(package):
    import importlib
    """Import the given plugins file from package"""
    return importlib.import_module(package, package=package)


def run_strategy(strategy_name):
    """ main script to run strategy based on file strategy name as input"""
    mod = _import(strategy_name)

    df, data_g, indicators0, colors0, sub_indicators0, sub_colors0, \
    colors, overlay, indicators, scatter, cash, margin, commission \
        = mod.get_signals()

    # activate functions used for strategy
    create_buy_order = mod.create_buy_order
    create_sell_order = mod.create_sell_order
    create_close_on_trades = mod.create_close_on_trades
    is_long_close_signal = mod.is_long_close_signal
    # is_short_close_signal = mod.is_short_close_signal
    is_long_buy_signal = mod.is_long_buy_signal
    # is_short_sell_signal = mod.is_short_sell_signal

    # initiate preplot plot
    dfpl = df.copy()
    dfpl_plot = df[1000:1200].copy()
    from utilities import plot_visualization
    metric_figure, fig, bottom_figure = \
        plot_visualization(dfpl_plot, indicators0,
                           colors0, sub_indicators0, sub_colors0)

    # metric_figure.show()

    s_ind = {
        "colors": colors,
        "overlay": overlay,
        "indicators": indicators,
        "scatter": scatter
    }
    data_g['s_ind'] = s_ind

    _LOG = logging.getLogger(__name__)
    # run this to start logging
    logging.debug("Begin")
    _LOG.setLevel(logging.DEBUG)

    class Plotlines:
        pass

    class Teststrategy(Strategy):
        data_g = []
        mysize = []

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

        def next(self):
            super().next()

            today_idx = self.data._Data__i - 1
            _LOG.debug(f"current candle {today_idx}")

            # determine if close trades
            days_max_order_to_fulfill = self.data_g['days_max_order_to_fulfill']
            limit_order_duration_on_exchange(self, days_max_order_to_fulfill, today_idx)
            create_close_on_trades(self)

            if is_long_buy_signal(self) and len(self.trades) == 0:
                create_buy_order(self)

            elif is_short_sell_signal(self) and len(self.trades) == 0:

                create_sell_order(self)

    def run_backtest(dfpl, cash, margin, commission, metric_figure, fig, bottom_figure, data_g, dfpl_plot):
        # run the backtesting engine
        bt = Backtest(dfpl, Teststrategy, cash=cash, margin=margin, commission=commission)

        stat = bt.run(data_g=data_g)
        print(stat)
        print(stat._trades)
        bt.plot(plot_width=1500, smooth_equity=True, superimpose=True, reverse_indicators=False)

        # update pre_plot with result signals from backtest
        date_strt, date_end = dfpl_plot.index[0], dfpl_plot.index[-1]
        _trades_entry = []
        for idx, ele in enumerate(stat._trades['EntryTime']):
            # checking for date in range
            if ele >= date_strt and ele <= date_end:
                _trades_entry.append(idx)

        stat_cp = stat.copy()
        a_trades = stat_cp._trades.iloc[_trades_entry]
        metric_figure, fig, bottom_figure = update_plots_with_signals(a_trades, metric_figure, fig, bottom_figure)
        metric_figure.show()

    run_backtest(dfpl, cash, margin, commission, metric_figure, fig, bottom_figure, data_g, dfpl_plot)


if __name__ == "__main__":
    strategy_name = 'teststrategy'
    run_strategy(strategy_name)
