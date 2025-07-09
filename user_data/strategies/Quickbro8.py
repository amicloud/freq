# --- Do not remove these libs ---
from freqtrade.optimize.space import Dimension, SKDecimal
from pandas import DataFrame
import math
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, merge_informative_pair, IntParameter

class Quickbro8(IStrategy):
    """
    Quickbro strategy version 8:
    - Hyperoptable quadratic ROI table parameters (a, b, c for ax^2+bx+c)
    - Samples ROI at 10 equidistant points between the parabola's real roots
    - Hyperoptable lookback_days for rolling low/high on 1d data
    - Buy when price crosses above rolling low (no multiplier)
    - Sell when price crosses above rolling high (no multiplier)
    - Stoploss set to 20%
    """
    class HyperOpt:
        @staticmethod
        def roi_space() -> list[Dimension]:
            # a small, b shifts horizontally, c between 0 and 1
            return [
                SKDecimal(-0.01, 0.01, decimals=3, name='quad_a'),
                SKDecimal(0.0, 200.0, decimals=1, name='quad_b'),
                SKDecimal(0.01, 0.99, decimals=2, name='quad_c'),
            ]

        @staticmethod
        def generate_roi_table(params: dict) -> dict[int, float]:
            a = params['quad_a']
            b = params['quad_b']
            c = params['quad_c']
            disc = b * b - 4 * a * c
            if disc <= 0 or a == 0:
                return {0: float(c)}
            r1 = (-b - math.sqrt(disc)) / (2 * a)
            r2 = (-b + math.sqrt(disc)) / (2 * a)
            x_start, x_end = min(r1, r2), max(r1, r2)
            roi_table = {}
            for i in range(10):
                x = x_start + i * (x_end - x_start) / 9
                t = int(x * 60 * 24)
                if t < 0:
                    continue
                y = a * x * x + b * x + c
                roi_table[t] = float(y)
                print(roi_table)
            return roi_table

    # Hyperoptable lookback window (in days)
    lookback_days = IntParameter(1, 200, default=126, space='buy')

    # Stoploss set to 28%
    stoploss = -0.28

    # Primary and informative timeframes
    timeframe = '15m'
    inf_timeframe = '1d'

    # Sell settings
    minimal_roi = {
        "0": 0.01,
        "120": 0.02,
        "480": 0.05,
        "1440": 0.10
    }
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    # Startup candles
    startup_candle_count = 127
    process_only_new_candles = False

    def informative_pairs(self):
        # Fetch 1d candles for all whitelisted pairs
        return [(pair, self.inf_timeframe) for pair in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Compute rolling low/high for lookback_days on daily data
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)
        for lb in self.lookback_days.range:
            low_col = f"{lb}d-low"
            high_col = f"{lb}d-high"
            informative[low_col] = informative['close'].rolling(window=lb, min_periods=1).min()
            informative[high_col] = informative['close'].rolling(window=lb, min_periods=1).max()
        return merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Buy when price crosses above the rolling low
        lb = self.lookback_days.value
        low_col = f"{lb}d-low_{self.inf_timeframe}"
        dataframe.loc[
            qtpylib.crossed_above(dataframe['close'], dataframe[low_col]),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Sell when price crosses above the rolling high
        lb = self.lookback_days.value
        high_col = f"{lb}d-high_{self.inf_timeframe}"
        dataframe.loc[
            qtpylib.crossed_above(dataframe['close'], dataframe[high_col]),
            'sell'
        ] = 1
        return dataframe
