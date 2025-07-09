import talib as ta
from freqtrade.strategy import IStrategy, IntParameter

class RSIComparisonStrategyV1(IStrategy):
    """
    Demonstration strategy using RSI.
    Buy when RSI crosses above buy threshold, sell when RSI crosses below sell threshold.
    """
    minimal_roi = {}
    stoploss = -0.10
    timeframe = '15m'
    buy_rsi = IntParameter(25, 35, default=30, space='buy')
    sell_rsi = IntParameter(65, 75, default=70, space='sell')
    startup_candle_count = 35

    def populate_indicators(self, dataframe, metadata):
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe.loc[
            (dataframe['rsi'] > self.buy_rsi.value) & (dataframe['rsi'].shift(1) <= self.buy_rsi.value),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe.loc[
            (dataframe['rsi'] < self.sell_rsi.value) & (dataframe['rsi'].shift(1) >= self.sell_rsi.value),
            'exit_long'] = 1
        return dataframe
