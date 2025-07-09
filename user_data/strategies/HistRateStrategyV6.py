from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, informative
import pandas as pd

class HistRateStrategyV6(IStrategy):
    """
    HistRateStrategyV6: Historical win-rate vs stop-loss filter with EMA slope filter.
    - Corrected loop start and use of .iloc for robust indexing.
    """
    INTERFACE_VERSION = 3
    timeframe = '15m'
    startup_candle_count = 200

    # Hyperoptable parameters
    threshold_win = DecimalParameter(0.5, 1.0, default=0.6, decimals=2, space='buy')
    threshold_loss = DecimalParameter(0.0, 0.5, default=0.2, decimals=2, space='buy')
    slope_window = IntParameter(2, 50, default=10, space='buy')

    # Fee and profit threshold
    fee = 0.005  # 0.5% total
    profit_threshold = 0.02 + fee

    # Risk management
    max_open_trades = 10
    use_exit_signal = False
    stoploss = -0.05  # 5% static stoploss

    minimal_roi = {'0': 0.02}

    @informative('1h')
    def populate_indicators_1h(self, df_1h: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df_1h['ema_50'] = df_1h['close'].ewm(span=50, adjust=False).mean()
        return df_1h

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return df

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        n = self.startup_candle_count
        thr_win = float(self.threshold_win.value)
        thr_loss = float(self.threshold_loss.value)
        slope_n = int(self.slope_window.value)
        profit_thr = self.profit_threshold
        stop_pct = abs(self.stoploss)

        # Determine the index to start: need historical candles and slope lookback
        start_index = max(n, slope_n)

        signals = [0] * len(df)
        ema_series = df['ema_50_1h'].iloc  # positional series

        for i in range(start_index, len(df)):
            # Historical window
            window = df.iloc[i - n:i]
            entry_price = (df['open'].iloc[i] + df['close'].iloc[i]) / 2
            wins = (window['high'] >= entry_price * (1 + profit_thr)).sum()
            losses = (window['low'] <= entry_price * (1 - stop_pct)).sum()
            win_rate = wins / n
            loss_rate = losses / n

            # EMA slope filter
            ema_current = ema_series[i]
            ema_past = ema_series[i - slope_n]
            ema_ok = ema_current > ema_past

            if win_rate >= thr_win and loss_rate <= thr_loss and ema_ok:
                signals[i] = 1

        df['enter_long'] = signals
        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df['exit_long'] = 0
        return df
