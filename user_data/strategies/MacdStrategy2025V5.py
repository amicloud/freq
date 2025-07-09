
from freqtrade.strategy import IStrategy, IntParameter
import talib
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame

class MacdStrategy2025V5(IStrategy):
    """
    MACD strategy V5:
    - Entry when MACD crosses above signal line,
      MACD is below a hyperoptable threshold,
      and MACD histogram is positive and rising.
    - Exit when MACD crosses below signal line,
      MACD is above a hyperoptable threshold,
      and MACD histogram is negative and falling.
    """
    # Strategy parameters
    timeframe = '15m'
    minimal_roi = {}
    stoploss = -0.10
    trailing_stop = False
    startup_candle_count = 30

    # Hyperoptable thresholds
    buy_macd_threshold = IntParameter(-100, 100, default=-10, space='buy')
    sell_macd_threshold = IntParameter(-100, 100, default=10, space='sell')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate MACD using TA-Lib
        macd, macd_signal, macd_hist = talib.MACD(
            dataframe['close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        dataframe['macd'] = macd
        dataframe['macd_signal'] = macd_signal
        dataframe['macd_hist'] = macd_hist
        # Track previous histogram for momentum filter
        dataframe['macd_hist_prev'] = dataframe['macd_hist'].shift(1)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Entry conditions:
        # 1. MACD crosses above signal line
        # 2. MACD below buy threshold
        # 3. Histogram positive and rising
        cond = (
            qtpylib.crossed_above(dataframe['macd'], dataframe['macd_signal']) &
            (dataframe['macd'] < self.buy_macd_threshold.value) &
            (dataframe['macd_hist'] > 0) &
            (dataframe['macd_hist'] > dataframe['macd_hist_prev'])
        )
        dataframe.loc[cond, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit conditions:
        # 1. MACD crosses below signal line
        # 2. MACD above sell threshold
        # 3. Histogram negative and falling
        cond = (
            qtpylib.crossed_below(dataframe['macd'], dataframe['macd_signal']) &
            (dataframe['macd'] > self.sell_macd_threshold.value) &
            (dataframe['macd_hist'] < 0) &
            (dataframe['macd_hist'] < dataframe['macd_hist_prev'])
        )
        dataframe.loc[cond, 'exit_long'] = 1
        return dataframe
