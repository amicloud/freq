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


class RLBare(BaseReinforcementLearningModel):
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
            :param action: int = The action made by the agent for the current candle.
            :return: float = the reward to give to the agent for current step
            """
            # first, penalize if the action is not valid
            if not self._is_valid(action):
                return -1

            # Ensure previous pnl is 0 at start
            if self._last_trade_tick is None:
                self._previous_pnl = 0.0

            # Get trade duration for time-based rewards
            if self._last_trade_tick is not None:
                trade_duration = self._current_tick - self._last_trade_tick #type: ignore
            else:
                trade_duration = 0
            max_trade_duration = self.rl_config.get("max_trade_duration_candles", 100)

            # Initialize previous pnl and pnl change
            if trade_duration == 1:
                self._previous_pnl = 0.0
                pnl_change = 0
            else:
                # Calculate PnL change since last tick
                pnl_change = pnl - self._previous_pnl
                # Update previous PnL for next tick
                self._previous_pnl = pnl
                #logger.info(f"PnL change: {pnl_change}") 
            
            # Buying when neutral
            if action == Actions.Buy.value and self._position == Positions.Neutral:
                reward = 0

            # Holding a long position
            if action == Actions.Neutral.value and self._position == Positions.Long:
                reward = 0

            # Staying neutral not trading
            if action == Actions.Neutral.value and self._position == Positions.Neutral:
                reward = 0

            # Selling a long position
            if action == Actions.Sell.value and self._position == Positions.Long:
                reward = 0

            

            return reward 
