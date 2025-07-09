# pragma: no cover

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta
class GreenPercentStrategyV2(IStrategy):
    """
    Version 1: Enters long when the percentage of green candles over a lookback period
    exceeds a threshold. Exits when the percentage over a separate lookback exceeds
    another threshold. Long-only strategy on 5m timeframe.
    """

    # Strategy parameters
    timeframe = '5m'
    startup_candle_count: int = 201

    thresh = 60*24*7
    minimal_roi = {str(thresh):0.0}

    # Stoploss at 10%
    stoploss = -0.10

    # Hyperoptable parameters
    entry_lookback = IntParameter(1, 100, default=20, space='buy')
    entry_threshold = DecimalParameter(0.0, 1.0, default=0.6, decimals=2, space='buy')
    rsi_entry = DecimalParameter(25.0, 35.0, default=30.0, decimals=1, space='buy')

    exit_lookback = IntParameter(1, 100, default=10, space='sell')
    exit_threshold = DecimalParameter(0.0, 1.0, default=0.5, decimals=2, space='sell')
    rsi_entry = DecimalParameter(65.0, 75.0, default=70.0, decimals=1, space='sell')

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Mark green candles
        df['green'] = (df['close'] > df['open']).astype(int)
        df['rsi'] = ta.RSI(df, timeperiod=14)
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Calculate percent of green candles over entry_lookback
        df['green_pct_entry'] = df['green'].rolling(self.entry_lookback.value).mean()

        df['enter_long'] = 0
        
        df.loc[
            df['green_pct_entry'] < self.entry_threshold.value and 
            df['rsi'] < self.rsi_entry.value,
            'enter_long'
        ] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Calculate percent of green candles over exit_lookback
        df['green_pct_exit'] = df['green'].rolling(self.exit_lookback.value).mean()
        df['exit_long'] = 0

        df.loc[
            df['green_pct_exit'] > self.exit_threshold.value and
            df['rsi'] > self.rsi_exit.value,
            'exit_long'
        ] = 1

        return df
