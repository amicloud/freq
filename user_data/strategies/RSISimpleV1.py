
from freqtrade.strategy import IStrategy, IntParameter
import pandas as pd
import talib.abstract as ta

class RSISimpleV1(IStrategy):
    # Hyperoptable parameters
    rsi_buy_threshold = IntParameter(25, 35, default=30, space='buy')
    rsi_sell_threshold = IntParameter(65, 75, default=70, space='sell')

    # Minimal settings
    timeframe = '15m'
    stoploss = -0.10
    minimal_roi = {}
    exit_profit_only = False

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe
        cond = (
            (dataframe['rsi'] > self.rsi_buy_threshold.value) &
            (dataframe['rsi'].shift(1) <= self.rsi_buy_threshold.value)
        )
        dataframe.loc[cond, 'enter_long'] = 1
        dataframe.loc[cond, 'enter_tag'] = f"rsi_cross_{self.rsi_buy_threshold.value}"
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe
        cond = (
            (dataframe['rsi'] > self.rsi_sell_threshold.value) &
            (dataframe['rsi'].shift(1) <= self.rsi_sell_threshold.value)
        )
        dataframe.loc[cond, 'exit_long'] = 1
        dataframe.loc[cond, 'exit_tag'] = f"rsi_exit_{self.rsi_sell_threshold.value}"
        return dataframe
