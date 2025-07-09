import logging
from functools import reduce
import datetime
from datetime import timedelta
import talib.abstract as ta
from pandas import DataFrame, Series
from technical import qtpylib
from typing import Optional
from freqtrade.strategy.interface import IStrategy
from technical.pivots_points import pivots_points
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from scipy.signal import argrelextrema
import numpy as np
import pandas_ta as pta
import math

logger = logging.getLogger(__name__)

class Freqai9(IStrategy):

    position_adjustment_enable = False
    ret_tgt = 0.005
    stoploss = -0.10

    accuracy_scores = DataFrame()

    order_types = {
        "entry": "limit",
        "exit": "market",
        "emergency_exit": "market",
        "force_exit": "market",
        "force_entry": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 120,
    }

    max_entry_position_adjustment = 1

    max_dca_multiplier = 2

    minimal_roi = {"0": ret_tgt * 2.0, "600": 0.0, "2400": -1.0}

    process_only_new_candles = True

    can_short = False

    plot_config = {
        "main_plot": {},
        "subplots": {
            "extrema": {
                "&s-extrema": {
                    "color": "#f53580",
                    "type": "line"
                },
                "&s-minima_sort_threshold": {
                    "color": "#f66151",
                    "type": "line"
                },
                "&s-maxima_sort_threshold": {
                    "color": "#8ff0a4",
                    "type": "line"
                }
            },
            "range_est": {
                "&-s_max": {
                    "color": "#a29db9",
                    "type": "line"
                },
                "&-s_min": {
                    "color": "#ac7fc",
                    "type": "line"
                }
            },
            "truth": {
                "maxima-exit": {
                    "color": "#8ff0a4",
                    "type": "bar"
                },
                "minima-exit": {
                    "color": "#f66151",
                    "type": "bar"
                }
            },
            "reddit": {
                "%%-social_volume_reddit/bitcoin": {
                    "color": "#75e918"
                }
            },
            "nvt": {
                "%%-nvt_5min/bitcoin": {
                    "color": "#be2306"
                }
            }
        }
    }

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 4},
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2,
            },

        ]

    use_exit_signal = False
    startup_candle_count: int = 400

    trailing_stop = True
    trailing_stop_positive = ret_tgt / 2.0
    trailing_stop_positive_offset = ret_tgt
    trailing_only_offset_is_reached = True

    def feature_engineering_expand_all(self, dataframe, period, **kwargs):
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, window=period)
        dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=period)
        dataframe["%-er-period"] = pta.er(dataframe['close'], length=period)
        dataframe["%-rocr-period"] = ta.ROCR(dataframe, timeperiod=period)
        dataframe["%-cmf-period"] = chaikin_mf(dataframe, periods=period)
        dataframe["%-tcp-period"] = top_percent_change(dataframe, period)
        dataframe["%-cti-period"] = pta.cti(dataframe['close'], length=period)
        dataframe["%-chop-period"] = qtpylib.chopiness(dataframe, period)
        dataframe["%-linear-period"] = ta.LINEARREG_ANGLE(
            dataframe['close'], timeperiod=period)
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        dataframe["%-atrp-period"] = dataframe[f"%-atr-period"] / \
            dataframe['close'] * 1000
        dataframe['%-ema-period'] = ta.EMA(dataframe, timeperiod=period)
        dataframe['%-sma-period'] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-moving_median-period"] = dataframe["close"].rolling(period).median()
        dataframe["%-dist_median-period"] = dataframe["close"] - dataframe["%-moving_median-period"]
        dataframe['%-dist_ema-period'] = dataframe['close'] - dataframe['%-ema-period']
        dataframe['%-dist_sma-period'] = dataframe['close'] -  dataframe['%-sma-period']
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2)
        dataframe["%-bb_lowerband-period"] = bollinger["lower"]
        dataframe["%-bb_middleband-period"] = bollinger["mid"]
        dataframe["%-bb_upperband-period"] = bollinger["upper"]
        dataframe["%-bb_width-period"] = (dataframe["%-bb_upperband-period"] - dataframe["%-bb_lowerband-period"]) / dataframe["%-bb_middleband-period"]
        return dataframe

    def feature_engineering_expand_basic(self, dataframe, **kwargs):
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-obv"] = ta.OBV(dataframe)
        dataframe["%-ibs"] = ((dataframe['close'] - dataframe['low']) / (dataframe['high'] - dataframe['low']))
        macd = ta.MACD(dataframe)
        dataframe['%-macd'] = macd['macd']
        dataframe['%-macdsignal'] = macd['macdsignal']
        dataframe['%-macdhist'] = macd['macdhist']
        dataframe['%-dist_to_macdsignal'] = get_distance(
            dataframe['%-macd'], dataframe['%-macdsignal'])
        dataframe['%-dist_to_zerohist'] = get_distance(
            0, dataframe['%-macdhist'])
        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe['vwap_upperband'] = vwap_high
        dataframe['vwap_middleband'] = vwap
        dataframe['vwap_lowerband'] = vwap_low
        dataframe['%-vwap_width'] = ((dataframe['vwap_upperband'] -
                                     dataframe['vwap_lowerband']) / dataframe['vwap_middleband']) * 100
        #dataframe = dataframe.copy()
        dataframe['%-dist_to_vwap_upperband'] = get_distance(
            dataframe['close'], dataframe['vwap_upperband'])
        dataframe['%-dist_to_vwap_middleband'] = get_distance(
            dataframe['close'], dataframe['vwap_middleband'])
        dataframe['%-dist_to_vwap_lowerband'] = get_distance(
            dataframe['close'], dataframe['vwap_lowerband'])
        dataframe['%-tail'] = (dataframe['close'] - dataframe['low']).abs()
        dataframe['%-wick'] = (dataframe['high'] - dataframe['close']).abs()
        pp = pivots_points(dataframe)
        dataframe['pivot'] = pp['pivot']
        dataframe['r1'] = pp['r1']
        dataframe['s1'] = pp['s1']
        dataframe['r2'] = pp['r2']
        dataframe['s2'] = pp['s2']
        dataframe['r3'] = pp['r3']
        dataframe['s3'] = pp['s3']
        dataframe['rawclose'] = dataframe['close']
        dataframe['%-dist_to_r1'] = get_distance(
            dataframe['close'], dataframe['r1'])
        dataframe['%-dist_to_r2'] = get_distance(
            dataframe['close'], dataframe['r2'])
        dataframe['%-dist_to_r3'] = get_distance(
            dataframe['close'], dataframe['r3'])
        dataframe['%-dist_to_s1'] = get_distance(
            dataframe['close'], dataframe['s1'])
        dataframe['%-dist_to_s2'] = get_distance(
            dataframe['close'], dataframe['s2'])
        dataframe['%-dist_to_s3'] = get_distance(
            dataframe['close'], dataframe['s3'])
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_low"] = dataframe["low"]
        dataframe["%-raw_high"] = dataframe["high"]
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        return dataframe

    def feature_engineering_standard(self, dataframe, **kwargs):
        dataframe["day_of_week"] = (dataframe["date"].dt.dayofweek)
        dataframe["hour_of_day"] = (dataframe["date"].dt.hour)
        dataframe['day_of_week_norm'] = 2 * math.pi * \
            dataframe['day_of_week'] / dataframe['day_of_week'].max()
        dataframe['hour_of_day_norm'] = 2 * math.pi * \
            dataframe['hour_of_day'] / dataframe['hour_of_day'].max()
        """
        dataframe['%%-day_of_week_cos'] = np.cos(dataframe['day_of_week_norm'])
        dataframe['%%-hour_of_day_cos'] = np.cos(dataframe['hour_of_day_norm'])
        dataframe['%%-day_of_week_sin'] = np.sin(dataframe['day_of_week_norm'])
        dataframe['%%-hour_of_day_sin'] = np.sin(dataframe['hour_of_day_norm'])
        """
        return dataframe

    def populate_indicators(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Version 10: 
        1. &-max-increase: maximum return over the next label_period_candles
        2. &-dd-before-max: binary flag indicating if a 5% drawdown occurs before the max increase
        """
        kernel = self.freqai_info["feature_parameters"]["label_period_candles"]
        dataframe["&-max-increase"] = (dataframe["close"].shift(-kernel).rolling(kernel).max()/dataframe["close"]) - 1
        dataframe["&-max-decrease"] = (dataframe["close"].shift(-kernel).rolling(kernel).min()/dataframe["close"]) - 1
        #dataframe["&-mean"] = (dataframe["close"].shift(-kernel).rolling(kernel).mean()/dataframe["close"]) - 1

        return dataframe

    def populate_entry_trend(
        self, df: DataFrame, metadata: dict
    ) -> DataFrame:
        # Enter long when prediction is on, we expect >n% upside, and no 5% drop before peak
        enter_long_conditions = [
            (df["do_predict"] == 1),
            df["&-max-increase"] >= self.ret_tgt,
            df["&-max-decrease"] > self.stoploss/2.0,
            #df["&-mean"] > self.ret_tgt
        ]
        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions),
                ["enter_long", "enter_tag"]
            ] = (1, "long")
        return df

    def populate_exit_trend(
        self, df: DataFrame, metadata: dict
    ) -> DataFrame:
        # Exit long once a 5% drawdown has occurred before the peak
        exit_long_conditions = [
            #df["do_predict"] == 1,
            #df["&-dd-before-max"] == 1
        ]
        if exit_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_long_conditions),
                "exit_long"
            ] = 0

        # (Optional) Exit short logic, mirror entry if you implement shorts
        # exit_short_conditions = [
        #     df["position"] == -1,
        #     df["&-max-increase"] < -0.02,
        #     df["&-dd-before-max"] == 0
        # ]
        # if exit_short_conditions:
        #     df.loc[
        #         reduce(lambda x, y: x & y, exit_short_conditions),
        #         "exit_short"
        #     ] = 1

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

def top_percent_change(dataframe: DataFrame, length: int) -> float:
    """
    Percentage change of the current close from the range maximum Open price
    :param dataframe: DataFrame The original OHLC dataframe
    :param length: int The length to look back
    """
    if length == 0:
        return (dataframe['open'] - dataframe['close']) / dataframe['close']
    else:
        return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']


def chaikin_mf(df, periods=20):
    close = df['close']
    low = df['low']
    high = df['high']
    volume = df['volume']
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0)
    mfv *= volume
    cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()
    return Series(cmf, name='cmf')



def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df, window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']


def EWO(dataframe, sma_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.EMA(df, timeperiod=sma_length)
    sma2 = ta.EMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif


def get_distance(p1, p2):
    return abs((p1) - (p2))
