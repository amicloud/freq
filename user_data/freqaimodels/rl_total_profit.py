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


class RLTotalProfit(BaseReinforcementLearningModel):
    """Reinforcement learner with profit aware reward function."""

    def fit(self, data_dictionary: dict[str, Any], dk: FreqaiDataKitchen, **kwargs):
        logger.info("RLTotalProfit 1.0")
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
            """Reward shaped by trade profit and overall episode profit."""
            if self._done:
                logger.info(f"Episode profit: {self._total_profit}")

            if not self._is_valid(action):
                return -1.0

            pnl = self.get_unrealized_profit()
            if self._last_trade_tick is not None:
                trade_duration = self._current_tick - self._last_trade_tick  # type: ignore
            else:
                trade_duration = 0
            max_trade_duration = self.rl_config.get("max_trade_duration_candles", 100)

            if trade_duration <= 1:
                pnl_change = 0.0
                self._previous_pnl = pnl
            else:
                pnl_change = pnl - self._previous_pnl
                self._previous_pnl = pnl

            reward = -self.fee

            if action == Actions.Buy.value and self._position == Positions.Neutral:
                # small penalty for opening a trade (covers fee)
                reward -= self.fee

            elif action == Actions.Neutral.value and self._position == Positions.Long:
                # reward improvements while holding
                reward += pnl_change * 10
                if pnl < 0:
                    reward += pnl
                reward -= (trade_duration / max_trade_duration) * 0.1

            elif action == Actions.Neutral.value and self._position == Positions.Neutral:
                # small penalty for inactivity
                reward -= 0.01

            elif action == Actions.Sell.value and self._position == Positions.Long:
                reward += pnl * 100
                if pnl > 0:
                    reward += 2.0
                reward -= self.fee
                reward -= (trade_duration / max_trade_duration) * 0.1

            # encourage improving episode profit
            reward += (self._total_profit - 1.0) / 5

            return tanh(2 * reward)
