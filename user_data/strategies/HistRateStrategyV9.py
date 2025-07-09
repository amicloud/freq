from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, informative
import pandas as pd

class HistRateStrategyV9(IStrategy):
    """
    HistRateStrategyV9: Historical win-rate vs stop-loss filter with EMA slope filter.
    - Loops over entire dataframe, uses startup_candle_count guard
    - Hyperoptable lookback_window for count, separate from startup_candle_count
    - Hyperoptable win/loss thresholds & EMA slope window
    - Static 5% stoploss, 2% ROI
    """
    INTERFACE_VERSION = 3
    timeframe = '15m'

    startup_candle_count = 200

    lookback_window = IntParameter(10, 200, default=200, space='buy')
    threshold_win   = DecimalParameter(0.5, 1.0, default=0.6, decimals=2, space='buy')
    threshold_loss  = DecimalParameter(0.0, 0.5, default=0.2, decimals=2, space='buy')
    slope_window    = IntParameter(2, 50, default=10, decimals=0, space='buy')

    fee              = 0.005
    profit_threshold = 0.02 + fee

    max_open_trades = 10
    use_exit_signal = False
    stoploss        = -0.05

    @property
    def minimal_roi(self) -> dict:
        return {0: 0.02}

    @informative('1h')
    def populate_indicators_1h(self, df_1h: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df_1h['ema_50'] = df_1h['close'].ewm(span=50, adjust=False).mean()
        return df_1h

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return df

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        lb       = int(self.lookback_window.value)
        thr_win  = float(self.threshold_win.value)
        thr_loss = float(self.threshold_loss.value)
        slope_n  = int(self.slope_window.value)
        profit_thr = self.profit_threshold
        stop_pct   = abs(self.stoploss)

        ema_series = df['ema_50_1h']
        signals    = [0] * len(df)

        for i in range(len(df)):
            if i < max(self.startup_candle_count, lb, slope_n):
                continue

            entry_price = (df['open'].iloc[i] + df['close'].iloc[i]) / 2

            wins, losses = 0, 0
            for j in range(lb):
                idx = i - 1 - j
                cand = df.iloc[idx]
                if cand['high'] >= entry_price * (1 + profit_thr):
                    wins += 1
                if cand['low']  <= entry_price * (1 - stop_pct):
                    losses += 1
            win_rate  = wins / lb
            loss_rate = losses / lb

            ema_current = ema_series.iloc[i]
            ema_past    = ema_series.iloc[i - slope_n]
            ema_ok      = ema_current > ema_past

            if win_rate >= thr_win and loss_rate <= thr_loss and ema_ok:
                signals[i] = 1

        df['enter_long'] = signals
        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df['exit_long'] = 0
        return df
