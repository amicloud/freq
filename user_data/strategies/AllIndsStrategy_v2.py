from freqtrade.strategy.interface import IStrategy
import talib.abstract as ta
import numpy as np
from pandas import DataFrame

class AllIndsStrategy_v2(IStrategy):
    """A strategy computing many TA indicators and using a random 50/50 entry/exit logic."""

    timeframe = '5m'
    minimal_roi = {}
    stoploss = -0.01
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Trend Indicators
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['sma10'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)

        # Momentum Indicators
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = ta.MACD(dataframe)
        dataframe['stoch_k'], dataframe['stoch_d'] = ta.STOCH(dataframe)
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['cci'] = ta.CCI(dataframe)
        dataframe['aroon_up'], dataframe['aroon_down'] = ta.AROON(dataframe)
        dataframe['willr'] = ta.WILLR(dataframe)

        # Volatility Indicators
        dataframe['atr'] = ta.ATR(dataframe)
        upper, middle, lower = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe['bb_upper'] = upper
        dataframe['bb_mid'] = middle
        dataframe['bb_lower'] = lower
        dataframe['sar'] = ta.SAR(dataframe)

        # Volume Indicators
        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['ad'] = ta.AD(dataframe)
        dataframe['mfi'] = ta.MFI(dataframe)

        # Additional Indicators
        dataframe['roc'] = ta.ROC(dataframe)
        dataframe['trix'] = ta.TRIX(dataframe, timeperiod=14)
        dataframe['mom'] = ta.MOM(dataframe)

        # Random signal for entry/exit
        dataframe['rand'] = np.random.random(size=len(dataframe))
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe['rand'] < 0.5, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe['rand'] >= 0.5, 'exit_long'] = 1
        return dataframe
