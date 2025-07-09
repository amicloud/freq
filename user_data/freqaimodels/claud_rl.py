import logging
from pathlib import Path
from typing import Any

import torch as th
from stable_baselines3.common.callbacks import ProgressBarCallback

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.Base3ActionRLEnv import Actions, Base3ActionRLEnv, Positions
from freqtrade.freqai.RL.BaseEnvironment import BaseEnvironment
from freqtrade.freqai.RL.BaseReinforcementLearningModel import BaseReinforcementLearningModel


logger = logging.getLogger(__name__)


class ClaudRL(BaseReinforcementLearningModel):
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
        logger.info("Simple RL 3.0 - Improved Reward Function")
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
            Improved reward function for 3-action long-only trading.
            
            Actions: Buy, Sell, Neutral
            Positions: Neutral, Long
            
            :param action: int = The action made by the agent for the current candle.
            :return: float = the reward to give to the agent for current step
            """
            pnl = self.get_unrealized_profit()
            
            # Get trade duration for time-based rewards
            if self._last_trade_tick is not None:
                trade_duration = self._current_tick - self._last_trade_tick #type: ignore
            else:
                trade_duration = 0
            max_trade_duration = self.rl_config.get("max_trade_duration_candles", 300)
            
            # === BUY ACTION ===
            if action == Actions.Buy.value:
                if self._position == Positions.Neutral:
                    # Reward for entering a long position
                    return 0.5
                elif self._position == Positions.Long:
                    # Small penalty for trying to buy when already long
                    return -0.1
            
            # === SELL ACTION ===
            elif action == Actions.Sell.value:
                if self._position == Positions.Long:
                    # Main reward logic - based on PnL
                    base_reward = pnl * 100.0  # Scale PnL to reasonable range
                    
                    # Time-based bonus/penalty
                    time_factor = 1.0
                    if trade_duration <= max_trade_duration * 0.3:
                        # Quick trades - bonus for profits, less penalty for losses
                        time_factor = 1.2 if pnl > 0 else 0.8
                    elif trade_duration > max_trade_duration:
                        # Long trades - encourage exit
                        time_factor = 0.8
                    
                    # Profit target bonus
                    profit_bonus = 0.0
                    if hasattr(self, 'profit_aim') and hasattr(self, 'rr'):
                        if pnl > self.profit_aim * self.rr:
                            profit_bonus = 2.0  # Bonus for hitting profit target
                    
                    final_reward = base_reward * time_factor + profit_bonus
                    
                    # Clamp extreme values
                    final_reward = max(min(final_reward, 50.0), -20.0)
                    
                    return final_reward
                    
                elif self._position == Positions.Neutral:
                    # Small penalty for trying to sell when not holding
                    return -0.1
            
            # === NEUTRAL ACTION ===
            elif action == Actions.Neutral.value:
                if self._position == Positions.Neutral:
                    # Small penalty for staying neutral to encourage trading
                    return -0.02
                    
                elif self._position == Positions.Long:
                    # Holding logic based on current PnL
                    if pnl > 0:
                        # Small reward for holding profitable positions
                        holding_reward = min(pnl * 5.0, 0.5)
                        
                        # Reduce reward for very long holds to encourage profit-taking
                        if trade_duration > max_trade_duration * 0.8:
                            holding_reward *= 0.5
                            
                        return holding_reward
                    else:
                        # Penalty for holding losing positions, increasing over time
                        if trade_duration < 12:
                            return 0.05 # Patience reward
                        loss_penalty = max(pnl * 10.0, -1.0)
                        
                        # Additional time penalty for long losing trades
                        if trade_duration > max_trade_duration * 0.5:
                            loss_penalty -= 0.5
                            
                        return loss_penalty
            
            # Default case
            return 0.0
