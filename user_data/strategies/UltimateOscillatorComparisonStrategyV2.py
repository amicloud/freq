import talib as ta
from freqtrade.strategy import IStrategy, IntParameter

class UltimateOscillatorComparisonStrategyV2(IStrategy):
    """
    Demonstration strategy using Ultimate Oscillator.
    Buy when UO crosses above buy threshold, sell when UO crosses below sell threshold.
    """
    minimal_roi = {}
    stoploss = -0.10
    timeframe = '15m'
    buy_uo = IntParameter(25, 35, default=30, space='buy')
    sell_uo = IntParameter(65, 75, default=70, space='sell')
    startup_candle_count = 35

    def populate_indicators(self, dataframe, metadata):
        # Pass high, low, close series to TA-Lib ULTOSC
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
        dataframe.loc[
            (dataframe['uo'] > self.buy_uo.value) &
            (dataframe['uo'].shift(1) <= self.buy_uo.value),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe.loc[
            (dataframe['uo'] < self.sell_uo.value) &
            (dataframe['uo'].shift(1) >= self.sell_uo.value),
            'exit_long'
        ] = 1
        return dataframe
