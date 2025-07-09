# --- Do not remove these libs ---
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from pandas import DataFrame
from functools import reduce
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, merge_informative_pair, IntParameter, DecimalParameter

class Quickbro4(IStrategy):
    """
    Quickbro strategy version 4:
    - Hyperoptable lookback_days for rolling median window
    - Hyperoptable median_mult_buy to buy when price crosses a percentage of the rolling median
    - Hyperoptable median_mult_sell to sell when price crosses a percentage of the rolling median
    - Maintains original ROI and stoploss logic
    """
    lookback_days = IntParameter(1, 200, default=126, space='buy')
    median_mult_buy = DecimalParameter(0.5, 0.99, decimals=2, default=1.0, space='buy')
    median_mult_sell = DecimalParameter(1.0, 2.0, decimals=2, default=1.0, space='sell')

    minimal_roi = {
        "0": 0.01,
        "120": 0.02,
        "480": 0.05,
        "1440": 0.10
    }

    stoploss = -0.10

    timeframe = '15m'
    inf_timeframe = '1d'

    use_sell_signal = False
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

    startup_candle_count = 127
    process_only_new_candles = False
    max_open_trades = 10
    def informative_pairs(self):
        """
        Define informative pairs to fetch 1d data for each whitelisted symbol
        """
        return [(pair, self.inf_timeframe) for pair in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Compute rolling median for all lookback_days values on daily data
        """
        informative = self.dp.get_pair_dataframe(
            pair=metadata['pair'],
            timeframe=self.inf_timeframe
        )

        for lb in self.lookback_days.range:
            col_med = f"{lb}d-med"
            informative[col_med] = informative['close'].rolling(window=lb, min_periods=1).median()

        dataframe = merge_informative_pair(
            dataframe, informative,
            self.timeframe, self.inf_timeframe,
            ffill=True
        )
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Buy when price crosses above median * median_mult_buy
        """
        lb = self.lookback_days.value
        med_col = f"{lb}d-med_{self.inf_timeframe}"
        mult = self.median_mult_buy.value
        dataframe.loc[
            qtpylib.crossed_above(dataframe['close'], dataframe[med_col] * mult),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Sell when price crosses above median * median_mult_sell
        """
        lb = self.lookback_days.value
        med_col = f"{lb}d-med_{self.inf_timeframe}"
        mult = self.median_mult_sell.value
        dataframe.loc[
            qtpylib.crossed_above(dataframe['close'], dataframe[med_col] * mult),
            'sell'
        ] = 1
        return dataframe
