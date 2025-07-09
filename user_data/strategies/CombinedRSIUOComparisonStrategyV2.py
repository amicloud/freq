import talib as ta
from freqtrade.strategy import IStrategy, IntParameter

class CombinedRSIUOComparisonStrategyV2(IStrategy):
    """
    Combined RSI and Ultimate Oscillator with bear market filter.
    Buy when both RSI and UO signals agree in bull or sideways markets.
    Sell when both RSI and UO exit signals agree.
    """
    minimal_roi = {}
    stoploss = -0.10
    timeframe = '15m'
    buy_rsi = IntParameter(25, 35, default=30, space='buy')
    sell_rsi = IntParameter(65, 75, default=70, space='sell')
    buy_uo = IntParameter(25, 35, default=30, space='buy')
    sell_uo = IntParameter(65, 75, default=70, space='sell')
    adx_threshold = IntParameter(20, 30, default=25, space='buy')
    startup_candle_count = 35

    def populate_indicators(self, dataframe, metadata):
        # RSI calculation
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        # Ultimate Oscillator calculation
        dataframe['uo'] = ta.ULTOSC(
            dataframe['high'],
            dataframe['low'],
            dataframe['close'],
            timeperiod1=7,
            timeperiod2=14,
            timeperiod3=28
        )
        # ADX and DI for trend filtering
        dataframe['adx'] = ta.ADX(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        # Entry when both RSI and UO cross above their buy thresholds...
        rsi_signal = (
            (dataframe['rsi'] > self.buy_rsi.value) &
            (dataframe['rsi'].shift(1) <= self.buy_rsi.value)
        )
        uo_signal = (
            (dataframe['uo'] > self.buy_uo.value) &
            (dataframe['uo'].shift(1) <= self.buy_uo.value)
        )
        # ...and market is bullish (+DI > -DI) or sideways (ADX below threshold)
        trend_filter = (
            (dataframe['plus_di'] > dataframe['minus_di']) |
            (dataframe['adx'] < self.adx_threshold.value)
        )
        dataframe.loc[rsi_signal & uo_signal & trend_filter, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        # Exit when both RSI and UO cross below their sell thresholds
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
