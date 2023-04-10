import inspect
import os
import sys
import time
import unittest

import _unittests
import warnings
from concurrent.futures.process import ProcessPoolExecutor
from contextlib import contextmanager
from glob import glob
from runpy import run_path
from tempfile import NamedTemporaryFile, gettempdir
# from unittests import TestCase
# from unittests.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from backtesting import Backtest, Strategy
from backtesting._stats import compute_drawdown_duration_peaks
from backtesting._util import _Array, _as_str, _Indicator, try_
from backtesting.lib import (
    OHLCV_AGG,
    SignalStrategy,
    TrailingStrategy,
    barssince,
    compute_stats,
    cross,
    crossover,
    plot_heatmaps,
    quantile,
    random_ohlc_data,
    resample_apply,
)
from backtesting.test import EURUSD, GOOG, SMA
from strategies import *
SHORT_DATA = GOOG.iloc[:20]  # Short data for fast tests with no indicator lag


@contextmanager
def _tempfile():
    with NamedTemporaryFile(suffix='.html') as f:
        if sys.platform.startswith('win'):
            f.close()
        yield f.name


@contextmanager
def chdir(path):
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


class TestBacktest(unittest.TestCase):
    def test_run(self):
        from run_strategy import run_backtest
        strategies = []
        strategies= ["strategy_dca", "strategy_sw", "strategy_macd"]
        for strategi in strategies:
            run_backtest(strategi)




if __name__ == '__main__':
    warnings.filterwarnings('error')
    unittest.main()
