from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
import pandas as pd

class HistRateStrategyV1(IStrategy):
    """
    HistRateStrategyV1: Historical win-rate vs stop-loss filter.
    Look back N candles, measure percentage of profitable and stop-loss signals,
    enter long if criteria met.
    """
    # Strategy version and timeframe
    timeframe = '15m'
    startup_candle_count = IntParameter(1, 200, default=50, space='strategy')
    threshold_win = DecimalParameter(0.5, 1.0, default=0.6, decimals=2, space='strategy')
    threshold_loss = DecimalParameter(0.0, 0.5, default=0.2, decimals=2, space='strategy')
    stoploss_pct = DecimalParameter(0.01, 0.10, default=0.05, decimals=2, space='sell')

    # Fee and profit threshold
    fee = 0.005  # 0.5% total
    profit_threshold = 0.02 + fee  # 2% profit + fee

    # Risk management
    max_open_trades = 10
    use_exit_signal = False
    stoploss = -0.99  # disable built-in stoploss; use custom_stoploss

    @property
    def minimal_roi(self) -> dict:
        # Fixed ROI table: 2% instant
        return {0: 0.02}

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs) -> float:
        # Dynamic stop-loss based on hyperoptable parameter
        return -float(self.stoploss_pct.value)

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        n = int(self.startup_candle_count.value)
        thr_win = float(self.threshold_win.value)
        thr_loss = float(self.threshold_loss.value)
        stop_p = float(self.stoploss_pct.value)
        profit_thr = self.profit_threshold

        signals = []
        for index in range(len(df)):
            if index < n:
                signals.append(0)
                continue
            window = df.iloc[index - n:index]
            entry_price = (df.at[index, 'open'] + df.at[index, 'close']) / 2
            wins = (window['high'] >= entry_price * (1 + profit_thr)).sum()
            losses = (window['low'] <= entry_price * (1 - stop_p)).sum()
            win_rate = wins / n
            loss_rate = losses / n
            signals.append(1 if (win_rate >= thr_win and loss_rate <= thr_loss) else 0)

        df['enter_long'] = signals
        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Exits handled by ROI and custom_stoploss
        df['exit_long'] = 0
        return df
