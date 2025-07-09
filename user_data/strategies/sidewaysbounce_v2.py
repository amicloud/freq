
# v2, 2025 Freqtrade conventions. Requires: freqtrade, ta-lib, qtpylib

from freqtrade.strategy import IStrategy
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
class SidewaysBounceV2(IStrategy):
    # Version 2 - 2025
    timeframe = "15m"
    can_short = False
    minimal_roi = {
        "0": 0.01,     # 2% at any time
        "30": 0.005,   # after 30min, 1.5%
        "120": 0.001,   # after 2 hours, 1%
        "360": 0.0001,  # after 6 hours, 0.5%
    }
    stoploss = -0.01
    trailing_stop = False
    trailing_stop_positive = 0.01    # 1% trailing
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = True
    ignore_buying_expired_candle_after = 10

    startup_candle_count = 30

    protections = [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 4  # 1 hour on 15m chart
        },
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 48,   # 12 hours
            "trade_limit": 3,
            "stop_duration_candles": 12,
            "only_per_pair": True
        },
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 96,  # 1 day
            "max_drawdown": 0.3,            # 30%
            "trade_limit": 1,
            "stop_duration_candles": 24,
            "only_per_pair": False,
        }
    ]

    # Optional: Filter low-liquidity pairs in pairlists.json

    def populate_indicators(self, dataframe, metadata):
        # Bollinger Bands
        upper, middle, lower  = ta.BBANDS(dataframe['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_lower'] = lower
        dataframe['bb_middle'] = middle
        dataframe['bb_upper'] = upper

        # RSI and ADX
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        dataframe['adx'] = ta.ADX(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)

        # ATR for optional volatility filter (not required for core logic)
        dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)

        # Volume filter (basic, can be extended)
        dataframe['vol_ma'] = dataframe['volume'].rolling(20).mean()

        # Shape & slope period
        window = 5
        # Concavity/Convexity: quadratic coefficient 'a'
        dataframe["bb_upper_quad"] = dataframe["bb_upper"].rolling(window).apply(lambda arr: np.polyfit(np.arange(len(arr)), arr, 2)[0], raw=True)
        dataframe["bb_lower_quad"] = dataframe["bb_lower"].rolling(window).apply(lambda arr: np.polyfit(np.arange(len(arr)), arr, 2)[0], raw=True) 
        # Boolean flags
        dataframe["bb_upper_convex_up"] = dataframe["bb_upper_quad"] > 0
        dataframe["bb_upper_concave_down"] = dataframe["bb_upper_quad"] < 0
        dataframe["bb_lower_convex_up"] = dataframe["bb_lower_quad"] > 0
        dataframe["bb_lower_concave_down"] = dataframe["bb_lower_quad"] < 0

        # Slopes: linear fit coefficient m
        dataframe["bb_upper_slope"] = dataframe["bb_upper"].rolling(window).apply(lambda arr: np.polyfit(np.arange(len(arr)), arr, 1)[0], raw=True)
        dataframe["bb_lower_slope"] = dataframe["bb_lower"].rolling(window).apply(lambda arr: np.polyfit(np.arange(len(arr)), arr, 1)[0], raw=True)

        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe.loc[
            (
                (dataframe['close'] <= dataframe['bb_lower']) &
                #(dataframe['rsi'] < 30) &
                (qtpylib.crossed_above(dataframe['rsi'], 28)) &
                (dataframe['adx'] < 25) &
                (dataframe['adx'] > -0) &
                (dataframe['volume'] > dataframe['vol_ma'])
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'bb_rsi_bounce')
        bbu = dataframe['bb_upper']
        dataframe.loc[
                (
                    (dataframe['bb_upper_concave_down'].rolling(5).sum() > 0 ) & 
                    (qtpylib.crossed_above(dataframe['bb_lower_slope'], 0)) &
                    (dataframe['adx'] > -0) 
                    #((bbu.shift(1) > bbu.shift(2)) & (bbu.shift(1) > bbu))
                ),
                ['enter_long', 'enter_tag']
        ] = (1, 'bb_upper_cc_lower_negative_slope')
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['rsi'], 70))
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'rsi_high')
        return dataframe
