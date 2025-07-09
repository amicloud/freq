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


class MyReinforcementLearner(BaseReinforcementLearningModel):
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
            Improved reward function for LONGS ONLY trading strategy.
            
            This reward function focuses on:
            - Encouraging profitable long entries at good timing
            - Rewarding profitable exits with bonus for beating profit targets
            - Penalizing excessive holding without profit
            - Encouraging timely exits to avoid drawdowns

            :param action: int = The action made by the agent for the current candle.
            :return: float = the reward to give to the agent for current step
            """
            # Penalize invalid actions
            if not self._is_valid(action):
                self.tensorboard_log("invalid_action", category="actions")
                return -5.0

            pnl = self.get_unrealized_profit()
            base_factor = 10.0
            
            # Configuration parameters
            max_trade_duration = self.rl_config.get("max_trade_duration_candles", 300)
            win_reward_factor = self.rl_config.get("model_reward_parameters", {}).get("win_reward_factor", 2000.0)
            
            # === LONG ENTRY LOGIC ===
            if action == Actions.Long_enter.value and self._position == Positions.Neutral:
                # Base reward for entering a long position
                entry_reward = 20.0
                
                # Bonus for entering when recent price action suggests uptrend
                # (This could be enhanced with technical indicators)
                self.tensorboard_log("long_entry", category="actions")
                return entry_reward
            
            # === NEUTRAL POSITION LOGIC ===
            if self._position == Positions.Neutral:
                if action == Actions.Neutral.value:
                    # Small penalty for staying neutral to encourage action
                    return -0.25
                else:
                    # No penalty for trying to enter when neutral
                    return 0.0
            
            # === LONG POSITION LOGIC ===
            if self._position == Positions.Long:
                trade_duration = self._current_tick - self._last_trade_tick
                
                # Calculate time-based factor
                if trade_duration <= max_trade_duration * 0.5:
                    # Optimal holding period
                    time_factor = 1.2
                elif trade_duration <= max_trade_duration:
                    # Acceptable holding period
                    time_factor = 1.0
                else:
                    # Too long, encourage exit
                    time_factor = 0.7
                
                if action == Actions.Long_exit.value:
                    # === EXIT REWARD CALCULATION ===
                    profit_factor = base_factor * time_factor
                    
                    # Strong bonus for profitable exits
                    if pnl > 0:
                        profit_factor *= 500.0
                        
                        # Extra bonus for beating profit target
                        if pnl > self.profit_aim * self.rr:
                            profit_factor *= win_reward_factor
                            self.tensorboard_log("profit_target_hit", category="rewards")
                    
                    # Moderate penalty for losing trades, but still better than holding
                    elif pnl < 0:
                        profit_factor *= 0.7
                        # Less penalty for quick losses (cut losses fast)
                        if trade_duration <= max_trade_duration * 0.3:
                            profit_factor *= 1.2
                    
                    self.tensorboard_log("long_exit", category="actions")
                    return float(pnl * profit_factor)
                
                elif action == Actions.Neutral.value:
                    # === HOLDING REWARD/PENALTY ===
                    if pnl > 0:
                        # Small positive reward for holding profitable positions
                        holding_reward = min(pnl * 10.0, 2.0)  # Cap the reward
                        
                        # But reduce reward as time goes on to encourage taking profits
                        if trade_duration > max_trade_duration * 0.7:
                            holding_reward *= 0.5
                        
                        return holding_reward
                    else:
                        # Increasing penalty for holding losing positions
                        loss_penalty = max(pnl * 50.0, -5.0)  # Cap the penalty
                        
                        # Increase penalty over time
                        time_penalty = trade_duration / max_trade_duration
                        
                        return loss_penalty - time_penalty
            
            # Default case
            return 0.0
