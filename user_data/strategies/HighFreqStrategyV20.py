# flake8: noqa: F401
# isort: skip_file
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import List, Tuple

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
)
import talib.abstract as ta

class HighFreqStrategyV20(IStrategy):
    """
    High-frequency, low-ROI strategy using EMA dip threshold and
    robust EMA-based trend prevention filter.
    """
    INTERFACE_VERSION = 3
    version = 20

    # Timeframe
    timeframe = '15m'

    # ROI and exit settings
    minimal_roi = {"0": 0.01}
    use_exit_signal = False
    stoploss = -0.2
    use_custom_stoploss = False

    # Ensure sufficient history
    startup_candle_count = 200

    # Hyperparameters
    buy_ema_short = IntParameter(10, 30, default=10, space='buy')
    buy_ema_long = IntParameter(40, 100, default=25, space='buy')
    buy_drop_pct = DecimalParameter(0.1, 30.0, decimals=1, default=1.0, space='buy')
    ema_trend_period = IntParameter(20, 50, default=25, space='buy')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMAs for trend and dip detection
        dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=self.buy_ema_short.value)
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=self.buy_ema_long.value)

        # EMA trend slope: require EMA long to have been rising over trend period
        slope = dataframe['ema_long'].diff()
        dataframe['ema_trend_ok'] = slope.rolling(self.ema_trend_period.value).min() > 0

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        cond = (
            (dataframe['ema_short'] > dataframe['ema_long']) &  # uptrend confirmation
            dataframe['ema_trend_ok'] &                          # EMA-based trend filter
            (dataframe['close'] < dataframe['ema_short'] * (1 - self.buy_drop_pct.value / 100))
        )
        dataframe.loc[cond, 'buy'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        return dataframe
