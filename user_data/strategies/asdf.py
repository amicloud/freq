from functools import reduce
from pandas import DataFrame
import numpy as np

class Freqai8:
    """
    Version 8: Updated freqai targets
    1. &-max-increase: maximum return over the next label_period_candles
    2. &-dd-before-max: binary flag indicating if a 5% drawdown occurs before the maximum increase
    """
    def __init__(self, freqai_info: dict):
        self.freqai_info = freqai_info

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        kernel = self.freqai_info["feature_parameters"]["label_period_candles"]
        close = dataframe["close"]

        # Target 1: max increase over next kernel candles
        dataframe["&-max-increase"] = (
            close.shift(-kernel)
            .rolling(kernel)
            .max()
            / close
            - 1
        )

        # Target 2: 5% drawdown before max increase
        def dd_before_max(row):
            idx = row.name
            future = close.iloc[idx + 1 : idx + 1 + kernel]
            if future.empty:
                return np.nan
            rel_pos = future.values.argmax()
            window = future.iloc[: rel_pos + 1]
            return int((window <= close.iloc[idx] * 0.95).any())

        dataframe["&-dd-before-max"] = dataframe.apply(dd_before_max, axis=1)

        return dataframe

    def populate_entry_trend(
        self, df: DataFrame, metadata: dict
    ) -> DataFrame:
        # Enter long when prediction is on, we expect >2% upside, and no 5% drop before peak
        enter_long_conditions = [
            df["do_predict"] == 1,
            df["&-max-increase"] > 0.02,
            df["&-dd-before-max"] == 0
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
            df["position"] == 1,
            df["&-dd-before-max"] == 1
        ]
        if exit_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_long_conditions),
                "exit_long"
            ] = 1

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
