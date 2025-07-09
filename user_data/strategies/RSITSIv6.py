import pandas as pd
import talib as ta
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter

class RSITSIv6(IStrategy):
    """
    RSI+TSI momentum strategy: separate buy and sell thresholds for RSI and TSI.
    """
    minimal_roi = {}
    stoploss = -0.1
    timeframe = '15m'

    # RSI hyperoptable thresholds
    buy_rsi = IntParameter(20, 40, default=30, space='buy')
    sell_rsi = IntParameter(60, 80, default=70, space='sell')
    # Separate TSI hyperoptable thresholds
    buy_tsi = DecimalParameter(-50.0, 0.0, decimals=2, default=0.00, space='buy')
    sell_tsi = DecimalParameter(0.0, 50.0, decimals=2, default=0.00, space='sell')

    startup_candle_count = 35

    def populate_indicators(self, dataframe, metadata):
        # RSI indicator
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        # Manual TSI calculation
        momentum = dataframe['close'].diff()
        ema1 = momentum.ewm(span=25, adjust=False).mean()
        ema2 = ema1.ewm(span=13, adjust=False).mean()
        abs_momentum = momentum.abs()
        abs_ema1 = abs_momentum.ewm(span=25, adjust=False).mean()
        abs_ema2 = abs_ema1.ewm(span=13, adjust=False).mean()
        dataframe['tsi'] = (ema2 / abs_ema2) * 100
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        # Entry when RSI crosses above buy_rsi AND TSI > buy_tsi
        rsi_cross = (
            (dataframe['rsi'] < self.buy_rsi.value) 
            & (dataframe['rsi'].shift(1) <= self.buy_rsi.value)
        )
        tsi_ok = dataframe['tsi'] < self.buy_tsi.value
        dataframe.loc[rsi_cross & tsi_ok, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        # Exit when RSI crosses below sell_rsi AND TSI < sell_tsi
        rsi_cross_down = (
            (dataframe['rsi'] > self.sell_rsi.value) 
            & (dataframe['rsi'].shift(1) >= self.sell_rsi.value)
        )
        tsi_bad = dataframe['tsi'] > self.sell_tsi.value
        dataframe.loc[rsi_cross_down & tsi_bad, 'exit_long'] = 1
        return dataframe
