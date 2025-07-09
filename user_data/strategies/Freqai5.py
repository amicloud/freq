import logging
from functools import reduce

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)


class Freqai5(IStrategy):

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
    # this is the maximum period fed to talib (timeframe independent)
    startup_candle_count: int = 400
    can_short = False
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:

        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["%-mom-period"] = ta.MOM(dataframe, timeperiod=period)

        # Patterns
        

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )

        #macd, macd_signal, macd_hist = ta.MACD(
        #    dataframe,
        #    fastperiod=int(period/2),
        #    slowperiod=period,
        #    signalperiod=int(period/3))

        #dataframe["m_macd-period"] = macd
        #dataframe["%-macd-period"] = dataframe["m_macd-period"]
        #dataframe["%-macd_signal-period"] = macd_signal
        #dataframe["%-macd_hist-period"] = macd_hist


        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]

        dataframe["%-bb_width-period"] = (
            dataframe["bb_upperband-period"] - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = dataframe["close"] / dataframe["bb_lowerband-period"]

        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)

        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        
        # Penetration parameter for candlestick patterns
        penetration = 0.01

        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        # Individual patterns for reference (keeping for debugging/analysis)
        # Strong Bullish Patterns
        dojistar = ta.CDLDOJISTAR(dataframe)
        engulfing = ta.CDLENGULFING(dataframe)
        whitesoldiers = ta.CDL3WHITESOLDIERS(dataframe)
        abandonedbaby = ta.CDLABANDONEDBABY(dataframe, penetration=penetration)
        breakaway = ta.CDLBREAKAWAY(dataframe)
        hammer = ta.CDLHAMMER(dataframe)
        invertedhammer = ta.CDLINVERTEDHAMMER(dataframe)
        ladderbottom = ta.CDLLADDERBOTTOM(dataframe)
        matchinglow = ta.CDLMATCHINGLOW(dataframe)
        mathold = ta.CDLMATHOLD(dataframe, penetration=penetration)
        morningdojistar = ta.CDLMORNINGDOJISTAR(dataframe, penetration=penetration)
        morningstar = ta.CDLMORNINGSTAR(dataframe, penetration=penetration)
        piercing = ta.CDLPIERCING(dataframe)
        risefall3methods = ta.CDLRISEFALL3METHODS(dataframe)
        tasukigap = ta.CDLTASUKIGAP(dataframe)
        unique3river = ta.CDLUNIQUE3RIVER(dataframe)
        
        # Strong Bearish Patterns
        twocrows = ta.CDL2CROWS(dataframe)
        blackcrows = ta.CDL3BLACKCROWS(dataframe)
        starsinsouth = ta.CDL3STARSINSOUTH(dataframe)
        advanceblock = ta.CDLADVANCEBLOCK(dataframe)
        darkcloudcover = ta.CDLDARKCLOUDCOVER(dataframe, penetration=penetration)
        eveningdojistar = ta.CDLEVENINGDOJISTAR(dataframe, penetration=penetration)
        eveningstar = ta.CDLEVENINGSTAR(dataframe, penetration=penetration)
        hangingman = ta.CDLHANGINGMAN(dataframe)
        shootingstar = ta.CDLSHOOTINGSTAR(dataframe)
        upsidegap2crows = ta.CDLUPSIDEGAP2CROWS(dataframe)
        
        # Indecision/Reversal Patterns
        doji = ta.CDLDOJI(dataframe)
        dragonflydoji = ta.CDLDRAGONFLYDOJI(dataframe)
        gravestonedoji = ta.CDLGRAVESTONEDOJI(dataframe)
        longleggeddoji = ta.CDLLONGLEGGEDDOJI(dataframe)
        rickshawman = ta.CDLRICKSHAWMAN(dataframe)
        spinningtop = ta.CDLSPINNINGTOP(dataframe)
        highwave = ta.CDLHIGHWAVE(dataframe)
        
        # Composite pattern group indicators
        dataframe["%-bullish_patterns"] = (
            (dojistar != 0) | 
            (engulfing > 0) |  # Engulfing can be +100 (bullish) or -100 (bearish)
            (whitesoldiers != 0) |
            (abandonedbaby != 0) |
            (breakaway != 0) |
            (hammer != 0) |
            (invertedhammer != 0) |
            (ladderbottom != 0) |
            (matchinglow != 0) |
            (mathold != 0) |
            (morningdojistar != 0) |
            (morningstar != 0) |
            (piercing != 0) |
            (risefall3methods > 0) |  # Can be +100 (rising) or -100 (falling)
            (tasukigap > 0) |  # Can be positive or negative
            (unique3river != 0)
        ).astype(int)
        
        dataframe["%-bearish_patterns"] = (
            (twocrows != 0) |
            (blackcrows != 0) |
            (starsinsouth != 0) |
            (advanceblock != 0) |
            (darkcloudcover != 0) |
            (eveningdojistar != 0) |
            (eveningstar != 0) |
            (hangingman != 0) |
            (shootingstar != 0) |
            (upsidegap2crows != 0) |
            (engulfing < 0) |  # Bearish engulfing
            (risefall3methods < 0) |  # Falling three methods
            (tasukigap < 0)  # Bearish tasuki gap
        ).astype(int)
        
        dataframe["%-indecision_patterns"] = (
            (doji != 0) |
            (dragonflydoji != 0) |
            (gravestonedoji != 0) |
            (longleggeddoji != 0) |
            (rickshawman != 0) |
            (spinningtop != 0) |
            (highwave != 0)
        ).astype(int)
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:

        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:

        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            .max()
            / dataframe["close"]
            - 1
        )
        dataframe["&-max-down"] = (dataframe["close"].shift(-self.freqai_info["feature_parameters"]["label_period_candles"]).rolling(self.freqai_info["feature_parameters"]["label_period_candles"]).min()/ dataframe["close"] - 1)

        # Classifiers are typically set up with strings as targets:
        # df['&s-up_or_down'] = np.where( df["close"].shift(-100) >
        #                                 df["close"], 'up', 'down')

        # If user wishes to use multiple targets, they can add more by
        # appending more columns with '&'. User should keep in mind that multi targets
        # requires a multioutput prediction model such as
        # freqai/prediction_models/CatboostRegressorMultiTarget.py,
        # freqtrade trade --freqaimodel CatboostRegressorMultiTarget

        # df["&-s_range"] = (
        #     df["close"]
        #     .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
        #     .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
        #     .max()
        #     -
        #     df["close"]
        #     .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
        #     .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
        #     .min()
        # )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # All indicators must be populated by feature_engineering_*() functions

        # the model will return all labels created by user in `set_freqai_targets()`
        # (& appended targets), an indication of whether or not the prediction should be accepted,
        # the target mean/std values for each of the labels created by user in
        # `set_freqai_targets()` for each training period.

        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_long_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] > 0.02,
            df["&-max-down"] > -0.025
        ]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")

        enter_short_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] < -0.01,
        ]

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
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
