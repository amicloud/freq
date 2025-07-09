
from freqtrade.strategy import IStrategy, IntParameter
import talib
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame

class RsiMacdStrategy2025V1(IStrategy):
    """
    Simple RSI strategy with MACD confirmation:
    - Entry when RSI crosses above a hyperoptable oversold threshold (buy_rsi)
      AND MACD is positive.
    - Exit when RSI crosses below a hyperoptable overbought threshold (sell_rsi)
      AND MACD is negative.
    """
    timeframe = '15m'
    minimal_roi = {}
    stoploss = -0.10
    trailing_stop = False
    startup_candle_count = 30

    # Hyperoptable RSI thresholds
    buy_rsi = IntParameter(10, 50, default=30, space='buy')
    sell_rsi = IntParameter(50, 90, default=70, space='sell')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate RSI
        dataframe['rsi'] = talib.RSI(dataframe['close'], timeperiod=14)
        # Calculate MACD for trend confirmation
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
        # Entry: RSI crosses above oversold threshold AND MACD > 0
        cond = (
            qtpylib.crossed_above(dataframe['rsi'], self.buy_rsi.value) &
            (dataframe['macd'] > 0)
        )
        dataframe.loc[cond, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit: RSI crosses below overbought threshold AND MACD < 0
        cond = (
            qtpylib.crossed_below(dataframe['rsi'], self.sell_rsi.value) 
            # &(dataframe['macd'] < 0)
        )
        dataframe.loc[cond, 'exit_long'] = 1
        return dataframe
