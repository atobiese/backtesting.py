import logging

from backtesting import Backtest
from backtesting import Strategy
from doc.examples.utilities import limit_order_duration_on_exchange, get_data, update_plots_with_signals

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
    metric_figure, fig, bottom_figure = \
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


    def _run_backtest(dfpl, init_data_g, metric_figure, fig,
                     bottom_figure, data_g, dfpl_plot):
        from strategies.strategy_sw_ny import Teststrategy
        # run the backtesting engine
        bt = Backtest(dfpl, Teststrategy,
                      exclusive_orders=init_data_g['exclusive_orders'],
                      cash=init_data_g['cash'], margin=init_data_g['margin'],
                      commission=init_data_g['commission'],
                      trade_on_close = init_data_g['trade_on_close'])

        stat = bt.run(data_g=data_g)

        print(stat)

        print(stat._trades)
        print(stat._strategy)
        bt.plot(plot_width=1500, smooth_equity=True,
                superimpose=True, reverse_indicators=False,
                open_browser=True, plot_drawdown=False)
        OPT_PARAMS = {'fast': range(2, 5, 2), 'slow': [2, 5, 7, 9]}
        # stat = bt.optimize(**OPT_PARAMS, data_g=data_g)
        import numpy as np
        # stat, heatmap, skopt_results = bt.optimize(
        #     fast=range(2, 20), slow=np.arange(2, 20, dtype=object),
        #     constraint=lambda p: p.fast < p.slow,
        #     max_tries=30,
        #     method='skopt',
        #     return_optimization=False,
        #     return_heatmap=True,
        #     random_state=2, **data_g)
        stat1 = bt.optimize(**OPT_PARAMS, extra_arg=data_g)
        print(stat1)
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
        metric_figure, fig, bottom_figure = update_plots_with_signals(a_trades, metric_figure, fig, bottom_figure)
        metric_figure.show()

    _run_backtest(dfpl, init_data_g, metric_figure,
                 fig, bottom_figure, data_g, dfpl_plot)


if __name__ == "__main__":
    # strategy_name = 'teststrategy'
    strategy_name = 'strategy_macd'
    # strategy_name = 'strategy_dca'
    strategy_name = 'strategy_sw_ny'
    # strategy_name = 'strategy_monthly'

    run_backtest(strategy_name)
