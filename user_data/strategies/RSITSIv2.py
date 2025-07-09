import talib as ta
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter

class RSITSIv2(IStrategy):
    """
    RSI+TSI momentum strategy: RSI trigger with TSI filter.
    """
    minimal_roi = {}
    stoploss = -0.10
    timeframe = '15m'

    buy_rsi = IntParameter(25, 35, default=30, space='buy')
    sell_rsi = IntParameter(65, 75, default=70, space='sell')
    tsi_filter = DecimalParameter(-1.0, 1.0, decimals=2, default=0.00, space='buy')

    startup_candle_count = 35

    def populate_indicators(self, dataframe, metadata):
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        dataframe['tsi'] = ta.TSI(dataframe['close'], timeperiod1=25, timeperiod2=13)
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
