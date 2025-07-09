import pandas as pd
import talib as ta
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter

class RSITSIv3(IStrategy):
    """
    RSI+TSI momentum strategy: RSI trigger with TSI filter (manual TSI calc).
    """
    minimal_roi = {}
    stoploss = -0.10
    timeframe = '15m'

    buy_rsi = IntParameter(25, 35, default=30, space='buy')
    sell_rsi = IntParameter(65, 75, default=70, space='sell')
    tsi_filter = DecimalParameter(-1.0, 1.0, decimals=2, default=0.00, space='buy')

    startup_candle_count = 35

    def populate_indicators(self, dataframe, metadata):
        # RSI indicator
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)

        # Manual True Strength Index (TSI) calculation
        momentum = dataframe['close'].diff()
        # First and second EMA smoothing
        ema1 = momentum.ewm(span=25, adjust=False).mean()
        ema2 = ema1.ewm(span=13, adjust=False).mean()
        # Absolute momentum smoothing
        abs_momentum = momentum.abs()
        abs_ema1 = abs_momentum.ewm(span=25, adjust=False).mean()
        abs_ema2 = abs_ema1.ewm(span=13, adjust=False).mean()
        # TSI = 100 * (smoothed momentum / smoothed absolute momentum)
        dataframe['tsi'] = (ema2 / abs_ema2) * 100

        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        rsi_cross = (
            (dataframe['rsi'] > self.buy_rsi.value) &
            (dataframe['rsi'].shift(1) <= self.buy_rsi.value)
        )
        tsi_ok = dataframe['tsi'] > self.tsi_filter.value
        dataframe.loc[rsi_cross & tsi_ok, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        rsi_cross_down = (
            (dataframe['rsi'] < self.sell_rsi.value) &
            (dataframe['rsi'].shift(1) >= self.sell_rsi.value)
        )
        tsi_bad = dataframe['tsi'] < self.tsi_filter.value
        dataframe.loc[rsi_cross_down & tsi_bad, 'exit_long'] = 1
        return dataframe
