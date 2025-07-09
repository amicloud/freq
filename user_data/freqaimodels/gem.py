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


class Gem(BaseReinforcementLearningModel):
   """
   This class provides a reinforcement learning model with a more
   sophisticated reward function.

   The user can override any other part of the IFreqaiModel tree.
   For example, the user can override `def fit()` or `def train()` or `def predict()`
   to take fine-tuned control over these processes.
   """

   def fit(self, data_dictionary: dict[str, Any], dk: FreqaiDataKitchen, **kwargs):
       """
       User customizable fit method
       :param data_dictionary: dict = common data dictionary containing all train/test features/labels/weights.
       :param dk: FreqaiDatakitchen = data kitchen for current pair.
       :return:
       model Any = trained model to be used for inference in dry/live/backtesting
       """
       logger.info("Starting training for RL model with custom reward function.")
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

   MyRLEnv: type[BaseEnvironment]

   class MyRLEnv(Base3ActionRLEnv):
       """
       Custom environment with a reward function designed to promote
       profitable and efficient trading behavior.
       """
       
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           # Initialize a variable to store the PnL from the previous step
           # This is crucial for calculating reward based on PnL change.
           self._previous_pnl = 0.0

       def calculate_reward(self, action: int) -> float:
           """
           Calculates the reward for the agent based on the action taken
           and the resulting market state.
           :param action: The action chosen by the agent.
           :return: The calculated reward.
           """
           # Hardcoded reward parameters
           profit_factor = 50.0
           win_bonus = 0.5
           loss_penalty = 0.5
           holding_pnl_factor = 20.0
           duration_penalty_factor = 0.05
           inaction_penalty = -0.001
           
           # --- 1. Strong penalty for invalid actions ---
           if not self._is_valid(action):
               return -1.0

           # --- 2. Get current state information ---
           pnl = self.get_unrealized_profit()
           trade_duration = self._current_tick - self._last_trade_tick if self._last_trade_tick is not None else 0
           max_trade_duration = self.rl_config.get("max_trade_duration_candles", 100)
           
           reward = 0.0

           # --- 3. Calculate reward based on action and position ---

           # A. Agent closes a long position
           if action == Actions.Sell.value and self._position == Positions.Long:
               # Reward is based on the final PnL of the trade
               # tanh is used to squash the reward to a range of [-1, 1] for training stability
               reward = tanh(pnl * profit_factor)
               
               # Add a clear bonus for winning or penalty for losing
               if pnl > 0:
                   reward += win_bonus
               else:
                   reward -= loss_penalty
               
               # Reset previous PnL for the next trade
               self._previous_pnl = 0.0

           # B. Agent holds a long position
           elif action == Actions.Neutral.value and self._position == Positions.Long:
               # Calculate the change in PnL since the last step
               pnl_change = pnl - self._previous_pnl
               
               # Reward the agent for positive changes in PnL
               reward = pnl_change * holding_pnl_factor
               
               # Apply a penalty that increases with trade duration
               # This encourages the agent to close trades in a timely manner
               duration_penalty = (trade_duration / max_trade_duration) * duration_penalty_factor
               reward -= duration_penalty
               
               # Update the previous PnL for the next step's calculation
               self._previous_pnl = pnl

           # C. Agent opens a new long position
           elif action == Actions.Buy.value and self._position == Positions.Neutral:
               # No immediate reward for buying. The reward comes from holding or selling profitably.
               # Reset previous PnL to start fresh for the new trade.
               self._previous_pnl = 0.0
               reward = 0.0

           # D. Agent stays neutral (is not in a trade)
           elif action == Actions.Neutral.value and self._position == Positions.Neutral:
               # Apply a small penalty for inaction to encourage the agent to seek opportunities.
               # This represents an "opportunity cost".
               reward = inaction_penalty

           return reward
