
from freqtrade.strategy import IStrategy, IntParameter
import talib
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame

class MacdStrategy2025V3(IStrategy):
    """
    Simple MACD strategy: buy when MACD crosses below a hyperoptable threshold; sell when MACD crosses above a hyperoptable threshold.
    Uses talib for MACD calculation and qtpylib for crossing signals.
    """
    # Strategy parameters
    timeframe = '15m'
    minimal_roi = {}
    stoploss = -0.20
    trailing_stop = False
    startup_candle_count = 30

    # Hyperoptable MACD thresholds
    buy_macd_threshold = IntParameter(-50, 0, default=-10, space='buy')
    sell_macd_threshold = IntParameter(0, 50, default=10, space='sell')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate MACD, signal, and histogram using talib
        macd, macd_signal, macd_hist = talib.MACD(
            dataframe['close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        dataframe['macd'] = macd
        dataframe['macd_signal'] = macd_signal
        dataframe['macd_hist'] = macd_hist
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            qtpylib.crossed_below(dataframe['macd'], self.buy_macd_threshold.value),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            qtpylib.crossed_above(dataframe['macd'], self.sell_macd_threshold.value),
            'exit_long'
        ] = 1
        return dataframe
