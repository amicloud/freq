# --- Do not remove these libs ---
from pandas import DataFrame
from functools import reduce
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, merge_informative_pair, IntParameter
# --------------------------------

class Quickbro1(IStrategy):
    # Hyperoptable lookback window (in days) for rolling min/max on 1d data
    lookback_days = IntParameter(1, 30, default=27, space='buy')

    # ROI targets (reduced for realistic gains)
    minimal_roi = {
        "0": 0.01,      # take 1% immediate profit
        "120": 0.02,    # after 2 hours, target 2%
        "480": 0.05,    # after 8 hours, target 5%
        "1440": 0.10    # after 24 hours, target 10%
    }

    # Stoploss
    stoploss = -0.99

    # Primary and informative timeframes
    timeframe = '15m'
    inf_timeframe = '1d'

    # Sell settings
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    # Startup candles: minimal to allow full informative DF merge
    startup_candle_count = 30
    process_only_new_candles = False

    def informative_pairs(self):
        """
        Define informative pairs to fetch 1d data for each whitelisted symbol
        """
        return [(pair, self.inf_timeframe) for pair in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Compute rolling low/high for all lookback_days values on daily data
        """
        # Fetch full 1d candle dataframe
        informative = self.dp.get_pair_dataframe(
            pair=metadata['pair'],
            timeframe=self.inf_timeframe
        )

        # Generate columns for all lookback_days values
        for lb in self.lookback_days.range:
            col_low = f"{lb}d-low"
            col_high = f"{lb}d-high"
            informative[col_low] = informative['close'].rolling(window=lb, min_periods=1).min()
            informative[col_high] = informative['close'].rolling(window=lb, min_periods=1).max()

        # Merge full informative DF; Freqtrade will suffix columns with '_1d'
        dataframe = merge_informative_pair(
            dataframe, informative,
            self.timeframe, self.inf_timeframe,
            ffill=True
        )
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Buy when price crosses above the rolling low for the chosen lookback_days
        """
        lb = self.lookback_days.value
        low_col = f"{lb}d-low_{self.inf_timeframe}"
        dataframe.loc[
            qtpylib.crossed_above(dataframe['close'], dataframe[low_col]),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Sell when price crosses above the rolling high for the chosen lookback_days
        """
        lb = self.lookback_days.value
        high_col = f"{lb}d-high_{self.inf_timeframe}"
        dataframe.loc[
            qtpylib.crossed_above(dataframe['close'], dataframe[high_col]),
            'sell'
        ] = 1
        return dataframe
