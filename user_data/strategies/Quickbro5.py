# --- Do not remove these libs ---
from freqtrade.optimize.space import Dimension, SKDecimal, Integer, Categorical
from pandas import DataFrame
from functools import reduce
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, merge_informative_pair, IntParameter, DecimalParameter

class Quickbro5(IStrategy):
    """
    Quickbro strategy version 5:
    - Hyperoptable lookback_days for rolling low/high on 1d data
    - Hyperoptable buy_mult to adjust buy threshold on rolling low
    - Hyperoptable sell_mult to adjust sell threshold on rolling high
    - Stoploss set to 20%
    - Maintains original ROI logic
    """
    # Hyperoptable lookback window (in days) for rolling low/high on 1d data
    lookback_days = IntParameter(1, 200, default=126, space='buy')
    # Multiplier for buy: applied to price before comparing to rolling low
    buy_mult = DecimalParameter(0.5, 1.5, decimals=2, default=1.0, space='buy')
    # Multiplier for sell: applied to price before comparing to rolling high
    sell_mult = DecimalParameter(0.5, 1.5, decimals=2, default=1.0, space='sell')

    # ROI targets
    minimal_roi = {
        "0": 0.01,
        "120": 0.02,
        "480": 0.05,
        "1440": 0.10
    }

    minimal_roi = {
        "0": 0.25
    } 

    # Stoploss set to 05%
    stoploss = -0.05

    # Primary and informative timeframes
    timeframe = '15m'
    inf_timeframe = '1d'

    # Sell settings
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Startup candles
    startup_candle_count = 200
    process_only_new_candles = False

    def informative_pairs(self):
        # Fetch 1d candles for all whitelisted pairs
        return [(pair, self.inf_timeframe) for pair in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Compute rolling low/high for all lookback_days values on daily data
        """
        # Fetch full 1d dataframe
        informative = self.dp.get_pair_dataframe(
            pair=metadata['pair'],
            timeframe=self.inf_timeframe
        )

        # Generate rolling low/high columns
        for lb in self.lookback_days.range:
            col_low = f"{lb}d-low"
            col_high = f"{lb}d-high"
            informative[col_low] = informative['low'].rolling(window=lb, min_periods=1).min()
            informative[col_high] = informative['close'].rolling(window=lb, min_periods=1).mean()

        # Merge into main timeframe
        dataframe = merge_informative_pair(
            dataframe, informative,
            self.timeframe, self.inf_timeframe,
            ffill=True
        )
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Buy when adjusted price crosses above the rolling low
        """
        lb = self.lookback_days.value
        low_col = f"{lb}d-low_{self.inf_timeframe}"
        mult = self.buy_mult.value
        dataframe.loc[
            qtpylib.crossed_above(dataframe['close'] * mult, dataframe[low_col]),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Sell when adjusted price crosses above the rolling high
        """
        lb = self.lookback_days.value
        high_col = f"{lb}d-high_{self.inf_timeframe}"
        mult = self.sell_mult.value
        dataframe.loc[
            qtpylib.crossed_above(dataframe['close'] * mult, dataframe[high_col]),
            'sell'
        ] = 1
        return dataframe
