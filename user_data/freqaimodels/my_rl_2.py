import logging
from pathlib import Path
from typing import Any

import torch as th
from stable_baselines3.common.callbacks import ProgressBarCallback

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions
from freqtrade.freqai.RL.BaseEnvironment import BaseEnvironment
from freqtrade.freqai.RL.BaseReinforcementLearningModel import BaseReinforcementLearningModel


logger = logging.getLogger(__name__)


class MyReinforcementLearner2(BaseReinforcementLearningModel):
    """
    Reinforcement Learning Model prediction model - LONGS ONLY VERSION.

    Users can inherit from this class to make their own RL model with custom
    environment/training controls. Define the file as follows:

    ```
    from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner

    class MyCoolRLModel(ReinforcementLearner):
    ```

    Save the file to `user_data/freqaimodels`, then run it with:

    freqtrade trade --freqaimodel MyCoolRLModel --config config.json --strategy SomeCoolStrat

    Here the users can override any of the functions
    available in the `IFreqaiModel` inheritance tree. Most importantly for RL, this
    is where the user overrides `MyRLEnv` (see below), to define custom
    `calculate_reward()` function, or to override any other parts of the environment.

    This class also allows users to override any other part of the IFreqaiModel tree.
    For example, the user can override `def fit()` or `def train()` or `def predict()`
    to take fine-tuned control over these processes.

    Another common override may be `def data_cleaning_predict()` where the user can
    take fine-tuned control over the data handling pipeline.
    """

    def fit(self, data_dictionary: dict[str, Any], dk: FreqaiDataKitchen, **kwargs):
        """
        User customizable fit method
        :param data_dictionary: dict = common data dictionary containing all train/test
            features/labels/weights.
        :param dk: FreqaiDatakitchen = data kitchen for current pair.
        :return:
        model Any = trained model to be used for inference in dry/live/backtesting
        """
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

    class MyRLEnv(Base5ActionRLEnv):  # type: ignore[no-redef]
        """
        User can override any function in BaseRLEnv and gym.Env. Here the user
        sets a custom reward based on profit and trade duration - LONGS ONLY VERSION.
        """

        def calculate_reward(self, action: int) -> float:
            """
            Balanced reward function for LONGS ONLY trading strategy.
            
            Designed to provide reasonable reward ranges and encourage good trading behavior
            without extreme penalties that can destabilize learning.

            :param action: int = The action made by the agent for the current candle.
            :return: float = the reward to give to the agent for current step
            """
            # Penalize invalid actions moderately
            if not self._is_valid(action):
                self.tensorboard_log("invalid_action", category="actions")
                return -1.0

            pnl = self.get_unrealized_profit()
            
            # Configuration parameters
            max_trade_duration = self.rl_config.get("max_trade_duration_candles", 300)
            win_reward_factor = self.rl_config.get("model_reward_parameters", {}).get("win_reward_factor", 200.0)
            
            # === LONG ENTRY LOGIC ===
            if action == Actions.Long_enter.value and self._position == Positions.Neutral:
                self.tensorboard_log("long_entry", category="actions")
                return 1.0  # Simple positive reward for entries
            
            # === NEUTRAL POSITION LOGIC ===
            if self._position == Positions.Neutral:
                if action == Actions.Neutral.value:
                    # Very small penalty for staying neutral
                    return -0.1
                else:
                    # Neutral reward for other actions when neutral
                    return 0.0
            
            # === LONG POSITION LOGIC ===
            if self._position == Positions.Long:
                trade_duration = self._current_tick - self._last_trade_tick
                
                if action == Actions.Long_exit.value:
                    # === EXIT REWARD CALCULATION ===
                    # Scale PnL to reasonable reward range
                    base_reward = pnl * 10.0  # Much smaller multiplier
                    
                    # Time-based adjustments
                    if trade_duration <= max_trade_duration * 0.3:
                        # Quick trades - slight bonus for profitable, less penalty for losses
                        time_bonus = 0.5 if pnl > 0 else 0.2
                    elif trade_duration <= max_trade_duration:
                        # Normal duration
                        time_bonus = 0.0
                    else:
                        # Too long - encourage exit
                        time_bonus = -0.5
                    
                    # Profit target bonus
                    target_bonus = 0.0
                    if pnl > 0 and pnl > self.profit_aim * self.rr:
                        target_bonus = 100.0 * win_reward_factor
                        self.tensorboard_log("profit_target_hit", category="rewards")
                    
                    final_reward = base_reward + time_bonus + target_bonus
                    
                    # Clamp the reward to prevent extreme values
                    final_reward = max(min(final_reward, 200.0), -200.0)
                    
                    self.tensorboard_log("long_exit", category="actions")
                    return float(final_reward)
                
                elif action == Actions.Neutral.value:
                    # === HOLDING REWARD/PENALTY ===
                    # Much gentler holding logic
                    if pnl > 0:
                        # Small reward for holding winners, decreasing over time
                        base_holding = min(pnl * 2.0, 0.5)  # Cap at 0.5
                        
                        if trade_duration > max_trade_duration * 0.8:
                            base_holding *= 0.5  # Reduce for long holds
                        
                        return base_holding
                    else:
                        # Gentle penalty for holding losers
                        loss_penalty = max(pnl * 5.0, -1.0)  # Cap at -1.0
                        
                        # Small additional penalty for very long losing trades
                        if trade_duration > max_trade_duration:
                            loss_penalty -= 0.5
                        
                        return loss_penalty
            
            # Default case
            return 0.0
