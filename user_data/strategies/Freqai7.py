import logging
from functools import reduce

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)


class Freqai7(IStrategy):

    minimal_roi = {"0": 0.05, "2400": -1}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "&-s_close": {"&-s_close": {"color": "blue"}},
            "do_predict": {
                "do_predict": {"color": "brown"},
            },
        },
    }

    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = False
    startup_candle_count: int = 400
    can_short = False
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:

        # Momentum Indicators
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mom-period"] = ta.MOM(dataframe, timeperiod=period)
        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)
        #slowk, slowd = ta.STOCH(
        #    dataframe,
        #   fastk_period=period,
        #   slowk_period=3,
        #   slowk_matype=0,
        #   slowd_period=3,
        #   slowd_matype=0,
        #)
        #dataframe["%-stoch_k-period"] = slowk
        #dataframe["%-stoch_d-period"] = slowd
        #stochrsi_k, stochrsi_d = ta.STOCHRSI(
        #    dataframe, timeperiod=period, fastk_period=3, fastd_period=3
        #)

        #dataframe["stochrsi_k"] = stochrsi_k
        #dataframe["stochrsi_d"] = stochrsi_d
        #dataframe["%-stochrsi_k-period"] = dataframe["stochrsi_k"]
        #dataframe["%-stochrsi_d-period"] = dataframe["stochrsi_d"]

        # Trend/Strength Indicators
              
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-plus_di-period"] = ta.PLUS_DI(dataframe, timeperiod=period)
        dataframe["%-minus_di-period"] = ta.MINUS_DI(dataframe, timeperiod=period)
        
        dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=period)
        ###dataframe["%-trix-period"] = ta.TRIX(dataframe, timeperiod=period)
        
        
        # Volatility Indicators
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=period, stds=2.2)
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]
        dataframe["%-bb_width-period"] = (dataframe["bb_upperband-period"] - dataframe["bb_lowerband-period"]) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = dataframe["close"] / dataframe["bb_lowerband-period"]
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        
        # Volume Indicators
        dataframe["%-obv-period"] = ta.OBV(dataframe)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-relative_volume-period"] = (dataframe["volume"] / dataframe["volume"].rolling(period).mean())
        dataframe["%-adosc-period"] = ta.ADOSC(dataframe, fastperiod=int(period/2), slowperiod=period)

        
        # Composite Oscillator Indicators
        dataframe["%-willr-period"] = ta.WILLR(dataframe, timeperiod=period)
        dataframe["%-ultosc-period"] = ta.ULTOSC(dataframe, timeperiod1=int(period/4), timeperiod2=int(period/2), timeperiod3=int(period))
        dataframe["%-cmo-period"] = ta.CMO(dataframe, timeperiod=period)

        # Overlap / Moving Averages
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        """
        macd = qtpylib.macd(dataframe,fast=int(period/5)+1, slow=int(period/3)+1, smooth=period)
        dataframe["%-macd-period"]        = macd['macd']
        dataframe["%-macd_signal-period"] = macd['signal']
        dataframe["%-macd_hist-period"]   = macd['histogram']
        """
        return dataframe
    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        penetration = 0.01
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            .max()
            / dataframe["close"]
            - 1
        )
        
        dataframe["&-max-down"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            .min()
            / dataframe["close"]
            - 1
        )
        
        return dataframe

    def populate_indicators(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_entry_trend(
        self, df: DataFrame, metadata: dict
    ) -> DataFrame:
        enter_long_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] > 0.02,
            df["&-max-down"] > -0.025,
        ]
        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions),
                ["enter_long", "enter_tag"],
            ] = (1, "long")
        return df

    def populate_exit_trend(
        self, df: DataFrame, metadata: dict
    ) -> DataFrame:
        exit_long_conditions = [df["do_predict"] == 1, df["&-s_close"] < 0]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1
        exit_short_conditions = [df["do_predict"] == 1, df["&-s_close"] > 0]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1
        return df

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()
        return True
        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False
        return True
