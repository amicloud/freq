# --- Do not remove these libs ---
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from pandas import DataFrame
from functools import reduce
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, merge_informative_pair, IntParameter
# --------------------------------

class Quickbro3FasterV5(IStrategy):
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

    lookback_days = IntParameter(1, 200, default=126, space='buy')

    minimal_roi = {
        "0": 0.01,
    }

    stoploss = -0.99

    timeframe = '15m'
    inf_timeframe = '1d'

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    startup_candle_count = 127
    process_only_new_candles = False

    def informative_pairs(self):
        return [(pair, self.inf_timeframe) for pair in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative = self.dp.get_pair_dataframe(
            pair=metadata['pair'],
            timeframe=self.inf_timeframe
        )

        # Rolling lows/highs
        for lb in self.lookback_days.range:
            informative[f"{lb}d-low"] = informative['close'].rolling(window=lb, min_periods=1).min()
            informative[f"{lb}d-high"] = informative['close'].rolling(window=lb, min_periods=1).max()

        # Daily indicators
        informative['rsi'] = qtpylib.rsi(informative['close'], window=14)
        macd_inf = qtpylib.macd(informative['close'])
        informative['macd']        = macd_inf['macd']
        informative['macd_signal'] = macd_inf['signal']
        informative['macd_hist']   = macd_inf['histogram']
        informative['ema_10'] = informative['close'].ewm(span=10, adjust=False).mean()
        informative['ema_50'] = informative['close'].ewm(span=50, adjust=False).mean()
        # Pivot Points
        pivot = (informative['high'] + informative['low'] + informative['close']) / 3
        informative['pivot'] = pivot
        informative['r1']    = 2 * pivot - informative['low']
        informative['s1']    = 2 * pivot - informative['high']

        dataframe = merge_informative_pair(
            dataframe, informative,
            self.timeframe, self.inf_timeframe,
            ffill=True
        )

        # Main 15m indicators
        dataframe['rsi'] = qtpylib.rsi(dataframe['close'], window=14)
        macd_main = qtpylib.macd(dataframe['close'])
        dataframe['macd']        = macd_main['macd']
        dataframe['macd_signal'] = macd_main['signal']
        dataframe['macd_hist']   = macd_main['histogram']
        dataframe['ema_10']      = dataframe['close'].ewm(span=10, adjust=False).mean()
        dataframe['ema_50']      = dataframe['close'].ewm(span=50, adjust=False).mean()

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        lb = self.lookback_days.value
        low_col = f"{lb}d-low_{self.inf_timeframe}"
        dataframe.loc[
            qtpylib.crossed_above(dataframe['close'], dataframe[low_col]),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        lb = self.lookback_days.value
        high_col = f"{lb}d-high_{self.inf_timeframe}"
        dataframe.loc[
            qtpylib.crossed_above(dataframe['close'], dataframe[high_col]),
            'sell'
        ] = 1
        return dataframe
