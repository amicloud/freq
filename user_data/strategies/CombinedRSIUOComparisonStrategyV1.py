import talib as ta
from freqtrade.strategy import IStrategy, IntParameter

class CombinedRSIUOComparisonStrategyV1(IStrategy):
    """
    Combined RSI and Ultimate Oscillator demonstration strategy.
    Buy when both RSI crosses above buy_rsi and UO crosses above buy_uo.
    Sell when both RSI crosses below sell_rsi and UO crosses below sell_uo.
    """
    minimal_roi = {}
    stoploss = -0.10
    timeframe = '15m'
    buy_rsi = IntParameter(25, 35, default=30, space='buy')
    sell_rsi = IntParameter(65, 75, default=70, space='sell')
    buy_uo = IntParameter(25, 35, default=30, space='buy')
    sell_uo = IntParameter(65, 75, default=70, space='sell')
    startup_candle_count = 35

    def populate_indicators(self, dataframe, metadata):
        # RSI indicator
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        # Ultimate Oscillator indicator
        dataframe['uo'] = ta.ULTOSC(
            dataframe['high'],
            dataframe['low'],
            dataframe['close'],
            timeperiod1=7,
            timeperiod2=14,
            timeperiod3=28
        )
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        rsi_signal = (
            (dataframe['rsi'] > self.buy_rsi.value) &
            (dataframe['rsi'].shift(1) <= self.buy_rsi.value)
        )
        uo_signal = (
            (dataframe['uo'] > self.buy_uo.value) &
            (dataframe['uo'].shift(1) <= self.buy_uo.value)
        )
        dataframe.loc[rsi_signal & uo_signal, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        rsi_exit = (
            (dataframe['rsi'] < self.sell_rsi.value) &
            (dataframe['rsi'].shift(1) >= self.sell_rsi.value)
        )
        uo_exit = (
            (dataframe['uo'] < self.sell_uo.value) &
            (dataframe['uo'].shift(1) >= self.sell_uo.value)
        )
        dataframe.loc[rsi_exit & uo_exit, 'exit_long'] = 1
        return dataframe
