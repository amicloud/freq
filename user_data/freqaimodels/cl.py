import logging
from pathlib import Path
from typing import Any
from math import tanh
import torch as th
from stable_baselines3.common.callbacks import ProgressBarCallback

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.Base3ActionRLEnv import Actions, Base3ActionRLEnv, Positions
from freqtrade.freqai.RL.BaseEnvironment import BaseEnvironment
from freqtrade.freqai.RL.BaseReinforcementLearningModel import BaseReinforcementLearningModel


logger = logging.getLogger(__name__)


class CL(BaseReinforcementLearningModel):
    """
    This class also allows users to override any other part of the IFreqaiModel tree.
    For example, the user can override `def fit()` or `def train()` or `def predict()` to take fine-tuned control over these processes.

    Another common override may be `def data_cleaning_predict()` where the user can take fine-tuned control over the data handling pipeline.
    """

    def fit(self, data_dictionary: dict[str, Any], dk: FreqaiDataKitchen, **kwargs):
        """
        User customizable fit method
        :param data_dictionary: dict = common data dictionary containing all train/test features/labels/weights.
        :param dk: FreqaiDatakitchen = data kitchen for current pair.
        :return:
        model Any = trained model to be used for inference in dry/live/backtesting
        """
        logger.info("RL Bare 0.01")
        train_df = data_dictionary["train_features"]
        total_timesteps = self.freqai_info["rl_config"]["train_cycles"] * len(train_df)

        policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=self.net_arch)

        if self.activate_tensorboard:
            tb_path = Path(dk.full_path / "tensorboard" / dk.pair.split("/")[0])
        else:
            tb_path = None

        if dk.pair not in self.dd.model_dictionary or not self.continual_learning:
            model = self.MODELCLASS(
                self.policy_type,
                self.train_env,
                policy_kwargs=policy_kwargs,
                tensorboard_log=tb_path,
                **self.freqai_info.get("model_training_parameters", {}),
            )
        else:
            logger.info(
                "Continual training activated - starting training from previously trained agent."
            )
            model = self.dd.model_dictionary[dk.pair]
            model.set_env(self.train_env)
        callbacks: list[Any] = [self.eval_callback, self.tensorboard_callback]
        progressbar_callback: ProgressBarCallback | None = None
        if self.rl_config.get("progress_bar", False):
            progressbar_callback = ProgressBarCallback()
            callbacks.insert(0, progressbar_callback)

        try:
            model.learn(
                total_timesteps=int(total_timesteps),
                callback=callbacks,
            )
        finally:
            if progressbar_callback:
                progressbar_callback.on_training_end()

        if Path(dk.data_path / "best_model.zip").is_file():
            logger.info("Callback found a best model.")
            best_model = self.MODELCLASS.load(dk.data_path / "best_model")
            return best_model

        logger.info("Couldn't find best model, using final model instead.")

        return model

    MyRLEnv: type[BaseEnvironment]  # type: ignore[assignment, unused-ignore]

    class MyRLEnv(Base3ActionRLEnv):  # type: ignore[no-redef]
        def calculate_reward(self, action: int) -> float:
            """
            Enhanced reward function that incentivizes profitable trades while penalizing losses
            and encouraging efficient trading behavior.
            
            :param action: int = The action made by the agent for the current candle.
            :return: float = the reward to give to the agent for current step
            """
            # First, penalize if the action is not valid
            if not self._is_valid(action):
                return -2.0
            
            # Get current PnL and price information
            pnl = self.get_unrealized_profit()
            current_price = self.prices.iloc[self._current_tick].open
            
            # Ensure previous pnl is 0 at start
            if self._last_trade_tick is None:
                self._previous_pnl = 0.0
            
            # Get trade duration for time-based penalties
            if self._last_trade_tick is not None:
                trade_duration = self._current_tick - self._last_trade_tick
            else:
                trade_duration = 0
            
            max_trade_duration = self.rl_config.get("max_trade_duration_candles", 100)
            
            # Calculate PnL change since last tick
            if trade_duration <= 1:
                self._previous_pnl = 0.0
                pnl_change = 0
            else:
                pnl_change = pnl - self._previous_pnl
                self._previous_pnl = pnl
            
            # Initialize base reward
            reward = 0.0
            
            # === ACTION-SPECIFIC REWARDS ===
            
            # Buying when neutral (entering position)
            if action == Actions.Buy.value and self._position == Positions.Neutral:
                # Small positive reward for taking action when neutral
                reward = 0.1
                
                # Bonus if we're entering at a good time (basic momentum check)
                if len(self.prices) > 5:
                    recent_prices = self.prices.iloc[max(0, self._current_tick-5):self._current_tick+1].close
                    if len(recent_prices) > 1:
                        momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
                        if momentum > 0.001:  # Positive momentum
                            reward += 0.2
            
            # Holding a long position
            elif action == Actions.Neutral.value and self._position == Positions.Long:
                # Reward based on PnL change
                if pnl_change > 0:
                    reward = min(pnl_change * 100, 2.0)  # Cap positive rewards
                elif pnl_change < 0:
                    reward = max(pnl_change * 100, -2.0)  # Cap negative penalties
                else:
                    reward = 0.0
                
                # Time-based penalty for holding too long
                if trade_duration > max_trade_duration * 0.7:
                    time_penalty = -0.1 * (trade_duration / max_trade_duration)
                    reward += time_penalty
                
                # Bonus for profitable positions
                if pnl > 0:
                    reward += 0.05
            
            # Staying neutral (not trading)
            elif action == Actions.Neutral.value and self._position == Positions.Neutral:
                # Small penalty for inaction to encourage trading
                reward = -0.02
                
                # But reward patience if market is choppy (high volatility)
                if len(self.prices) > 10:
                    recent_prices = self.prices.iloc[max(0, self._current_tick-10):self._current_tick+1].close
                    if len(recent_prices) > 1:
                        volatility = recent_prices.std() / recent_prices.mean()
                        if volatility > 0.02:  # High volatility threshold
                            reward = 0.05  # Reward for staying out of choppy markets
            
            # Selling a long position (closing position)
            elif action == Actions.Sell.value and self._position == Positions.Long:
                # Base reward heavily dependent on final PnL
                if pnl > 0:
                    # Profitable trade - strong positive reward
                    profit_reward = min(pnl * 200, 5.0)  # Scale and cap the reward
                    reward = profit_reward
                    
                    # Bonus for quick profitable trades
                    if trade_duration <= max_trade_duration * 0.3:
                        reward += 0.5
                    
                    # Additional bonus for high profit margins
                    if pnl > 0.02:  # 2% profit
                        reward += 1.0
                        
                elif pnl < 0:
                    # Losing trade - penalty but reward cutting losses
                    loss_penalty = max(pnl * 200, -3.0)  # Scale and cap the penalty
                    reward = loss_penalty
                    
                    # Smaller penalty if we cut losses quickly
                    if trade_duration <= max_trade_duration * 0.2:
                        reward += 0.3  # Reward for cutting losses fast
                    
                    # Severe penalty for large losses
                    if pnl < -0.05:  # -5% loss
                        reward -= 1.0
                        
                else:
                    # Breakeven trade
                    reward = 0.1  # Small reward for not losing money
            
            # === ADDITIONAL RISK MANAGEMENT REWARDS ===
            
            # Penalty for holding losing positions too long
            if self._position == Positions.Long and pnl < -0.02 and trade_duration > max_trade_duration * 0.5:
                reward -= 0.5
            
            # Reward for maintaining reasonable position sizing (if applicable)
            # This would need to be customized based on your position sizing logic
            
            # === FINAL REWARD ADJUSTMENTS ===
            
            # Apply tanh to smooth extreme rewards
            reward = tanh(reward)
            
            # Add small random noise to prevent overfitting to specific patterns
            import random
            noise = random.uniform(-0.01, 0.01)
            reward += noise
            
            return reward
