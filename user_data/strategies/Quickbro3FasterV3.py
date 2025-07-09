# --- Do not remove these libs ---
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from pandas import DataFrame
from functools import reduce
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, merge_informative_pair, IntParameter
# --------------------------------

class Quickbro3FasterV3(IStrategy):
    class HyperOpt:
        # Define custom ROI space with 0 lower bound and 2 decimal precision
        def roi_space() -> list[Dimension]:
            return [
                SKDecimal(0.00, 2, decimals=2, name='roi_t1'),
                SKDecimal(0.00, 5, decimals=2, name='roi_t2'),
                SKDecimal(0.00, 10, decimals=2, name='roi_t3'),
                SKDecimal(0.00, 20, decimals=2, name='roi_t4'),
                Integer(0, 90, name='roi_t5'),
                SKDecimal(0.01, 0.10, decimals=2, name='roi_p1'),
                SKDecimal(0.01, 0.10, decimals=2, name='roi_p2'),
                SKDecimal(0.01, 0.10, decimals=2, name='roi_p3'),
                SKDecimal(0.01, 0.10, decimals=2, name='roi_p4'),
            ]

        def generate_roi_table(params: dict) -> dict[int, float]:
            t1 = int(params['roi_t1'] * 60 * 24)
            t2 = int(params['roi_t2'] * 60 * 24)
            t3 = int(params['roi_t3'] * 60 * 24)
            t4 = int(params['roi_t4'] * 60 * 24)
            t5 = int(params['roi_t5'] * 60 * 24)
            p1 = params['roi_p1']
            p2 = params['roi_p2']
            p3 = params['roi_p3']
            p4 = params['roi_p4']

            roi_table = {}
            roi_table[t1] = p1
            roi_table[t1 + t2] = p1 + p2
            roi_table[t1 + t2 + t3] = p1 + p2 + p3
            roi_table[t1 + t2 + t3 + t4] = p1 + p2 + p3 + p4
            roi_table[t1 + t2 + t3 + t4 + t5] = 0.0

            return roi_table

    # Hyperoptable lookback window (in days) for rolling min/max on 1d data
    lookback_days = IntParameter(1, 200, default=126, space='buy')

    # ROI targets (reduced for realistic gains)
    minimal_roi = {
        "0": 0.01,      # take 1% immediate profit
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
    startup_candle_count = 127
    process_only_new_candles = False

    def informative_pairs(self):
        """
        Define informative pairs to fetch 1d data for each whitelisted symbol
        """
        return [(pair, self.inf_timeframe) for pair in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Compute indicators on both 15m and 1d timeframes
        """
        # Fetch full 1d candle dataframe
        informative = self.dp.get_pair_dataframe(
            pair=metadata['pair'],
            timeframe=self.inf_timeframe
        )

        # Generate rolling low/high for all lookback_days values
        for lb in self.lookback_days.range:
            col_low = f"{lb}d-low"
            col_high = f"{lb}d-high"
            informative[col_low] = informative['close'].rolling(window=lb, min_periods=1).min()
            informative[col_high] = informative['close'].rolling(window=lb, min_periods=1).max()

        # Daily indicators (1d)
        informative['rsi'] = qtpylib.rsi(informative['close'], window=14)
        macd_inf = qtpylib.macd(informative['close'])
        informative['macd'] = macd_inf['macd']
        informative['macd_signal'] = macd_inf['macdsignal']
        informative['macd_hist'] = macd_inf['macdhist']
        informative['ema_10'] = informative['close'].ewm(span=10, adjust=False).mean()
        informative['ema_50'] = informative['close'].ewm(span=50, adjust=False).mean()
        # Pivot Points Indicator (PPI) for daily timeframe
        pivot = (informative['high'] + informative['low'] + informative['close']) / 3
        informative['pivot'] = pivot
        informative['r1'] = 2 * pivot - informative['low']
        informative['s1'] = 2 * pivot - informative['high']

        # Merge informative into the main dataframe; suffixes with _1d
        dataframe = merge_informative_pair(
            dataframe, informative,
            self.timeframe, self.inf_timeframe,
            ffill=True
        )

        # Main timeframe indicators (15m)
        dataframe['rsi'] = qtpylib.rsi(dataframe['close'], window=14)
        macd = qtpylib.macd(dataframe['close'])
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        dataframe['macd_hist'] = macd['macdhist']
        dataframe['ema_10'] = dataframe['close'].ewm(span=10, adjust=False).mean()
        dataframe['ema_50'] = dataframe['close'].ewm(span=50, adjust=False).mean()

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
