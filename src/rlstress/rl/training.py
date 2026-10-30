"""PPO training with stable-baselines3.

* Shock model: fitted once on the training split and passed in; training and
  validation environments share it (validation only changes the anchor dates
  that set the observed market state).
* Training env: anchors from ``simulation.anchor_window`` (train).
* Observation normalisation: ``VecNormalize`` on observations only (rewards are
  not normalised, so reward components keep their meaning). Statistics are
  saved with every checkpoint and frozen for evaluation.
* Checkpoint selection: every ``eval_freq`` timesteps the deterministic policy is
  run on ``selection_episodes`` validation-anchor episodes (seeds disjoint from
  the reporting seeds). Best = highest valid-breach rate, ties broken by mean
  episode return.
* Reproducibility: ``set_random_seed(seed)``, single torch thread, CPU, one env.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rlstress.config import ProjectConfig
from rlstress.env import StressTestEnv, make_env
from rlstress.rl.evaluation import COMPONENTS, evaluate_policy, summarize
from rlstress.rl.policies import FrozenObsNormalizer, SB3Policy
from rlstress.shocks import ShockModel

logger = logging.getLogger(__name__)


def with_threshold(config: ProjectConfig, threshold: float, anchor_window: str | None = None) -> ProjectConfig:
    """Copy of ``config`` with ``simulation.min_plausibility`` (and optionally the anchor window) replaced."""
    sim = dataclasses.replace(config.simulation, min_plausibility=threshold,
                              anchor_window=anchor_window or config.simulation.anchor_window)
    return dataclasses.replace(config, simulation=sim)


def build_env(config: ProjectConfig, source: pd.DataFrame, model: ShockModel, threshold: float,
              anchor_window: str) -> StressTestEnv:
    """Environment with the train-fitted ``model``, a plausibility threshold and an anchor split."""
    return make_env(with_threshold(config, threshold, anchor_window), source, model)


def build_vec_env(config: ProjectConfig, source: pd.DataFrame, model: ShockModel,
                  threshold: float) -> DummyVecEnv | VecNormalize:
    """Single training env (train anchors) wrapped for SB3."""
    venv = DummyVecEnv([lambda: build_env(config, source, model, threshold, config.simulation.anchor_window)])
    if config.training.normalize_observations:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return venv


def ppo_kwargs(config: ProjectConfig) -> dict[str, Any]:
    """PPO constructor arguments from ``[training]``."""
    t = config.training
    return dict(learning_rate=t.learning_rate, n_steps=t.n_steps, batch_size=t.batch_size, n_epochs=t.n_epochs,
                gamma=t.gamma, gae_lambda=t.gae_lambda, clip_range=t.clip_range, ent_coef=t.ent_coef,
                vf_coef=t.vf_coef, max_grad_norm=t.max_grad_norm, device=t.device,
                policy_kwargs={"log_std_init": t.log_std_init, "net_arch": list(t.net_arch)})


class EpisodeLogger(BaseCallback):
    """Collects per-episode totals of every reward component and the termination reason."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, Any]] = []
        self._acc: dict[str, float] = dict.fromkeys(COMPONENTS, 0.0)
        self._len = 0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        for k, v in info.get("reward_components", {}).items():
            self._acc[k] += v
        self._len += 1
        if self.locals["dones"][0]:
            if info.get("invalid"):
                term = "invalid"
            elif info.get("valid_breach"):
                term = "valid_breach"
            elif info.get("implausible_breach"):
                term = "implausible_breach"
            else:
                term = "horizon"
            self.rows.append({"timesteps": self.num_timesteps, "length": self._len, "termination": term,
                              "total_reward": sum(self._acc.values()),
                              "path_drawdown": info.get("path_drawdown"),
                              "historical_percentile": info.get("path_historical_percentile"),
                              **{f"reward_{k}": v for k, v in self._acc.items()}})
            self._acc = dict.fromkeys(COMPONENTS, 0.0)
            self._len = 0
        return True


class SelectionCallback(BaseCallback):
    """Periodic validation-anchor evaluation; saves the best checkpoint and its normalisation stats."""

    def __init__(self, eval_env: StressTestEnv, seeds: list[int], eval_freq: int, out_dir: Path) -> None:
        super().__init__()
        self.eval_env = eval_env
        self.seeds = seeds
        self.eval_freq = eval_freq
        self.out_dir = out_dir
        self.best: tuple[float, float] | None = None
        self.log: list[dict[str, Any]] = []

    def _evaluate(self) -> None:
        venv = self.model.get_vec_normalize_env()
        normalizer = FrozenObsNormalizer.from_vec_normalize(venv) if venv is not None else None
        policy = SB3Policy(self.model, normalizer, deterministic=True)
        s = summarize(evaluate_policy(self.eval_env, policy, self.seeds))
        key = (s["valid_breach_rate"], s["episode_return"]["mean"])
        entry = {"timesteps": self.num_timesteps, "valid_breach_rate": key[0], "mean_return": key[1],
                 "breach_rate": s["breach_rate"], "implausible_breach_rate": s["implausible_breach_rate"]}
        if self.best is None or key > self.best:
            self.best = key
            self.model.save(self.out_dir / "best_model.zip")
            if venv is not None:
                venv.save(str(self.out_dir / "best_vecnormalize.pkl"))
            entry["new_best"] = True
        self.log.append(entry)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            self._evaluate()
        return True

    def _on_training_end(self) -> None:
        if not self.log or self.log[-1]["timesteps"] != self.num_timesteps:
            self._evaluate()


@dataclass
class TrainResult:
    """Paths and summary of one training run."""

    run_dir: Path
    seed: int
    threshold: float
    timesteps: int
    seconds: float
    selection_log: list[dict[str, Any]]
    best: dict[str, Any]


def vecnormalize_path(model_path: Path) -> Path:
    """Normalisation stats saved next to a model (``best_``/``final_`` or SB3 checkpoint naming)."""
    name = model_path.stem
    if name.endswith("_model"):
        return model_path.with_name(name.replace("_model", "_vecnormalize") + ".pkl")
    return model_path.with_name(name.replace("model_", "model_vecnormalize_", 1) + ".pkl")


def train_ppo(config: ProjectConfig, source: pd.DataFrame, model: ShockModel, threshold: float, seed: int,
              run_dir: Path, total_timesteps: int | None = None, resume_from: Path | None = None) -> TrainResult:
    """Train (or resume) PPO for one seed and threshold; write logs, checkpoints and the best model."""
    t = config.training
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(t.torch_threads)
    set_random_seed(seed)
    steps = total_timesteps or t.total_timesteps
    if resume_from is not None:
        venv = DummyVecEnv([lambda: build_env(config, source, model, threshold, config.simulation.anchor_window)])
        stats = vecnormalize_path(resume_from)
        if config.training.normalize_observations:
            venv = VecNormalize.load(str(stats), venv)
            venv.training = True
        ppo = PPO.load(resume_from, env=venv, device=t.device)
        reset = False
    else:
        venv = build_vec_env(config, source, model, threshold)
        ppo = PPO(t.policy, venv, seed=seed, verbose=0, **ppo_kwargs(config))
        reset = True
    ppo.set_logger(configure(str(run_dir), ["csv"]))
    eval_env = build_env(config, source, model, threshold, "validation")
    selection = SelectionCallback(eval_env, list(range(t.selection_seed, t.selection_seed + t.selection_episodes)),
                                  t.eval_freq, run_dir)
    episodes = EpisodeLogger()
    checkpoints = CheckpointCallback(t.checkpoint_freq, str(run_dir / "checkpoints"), name_prefix="model",
                                     save_vecnormalize=config.training.normalize_observations)
    start = time.time()
    ppo.learn(steps, callback=CallbackList([episodes, selection, checkpoints]), reset_num_timesteps=reset)
    seconds = time.time() - start
    ppo.save(run_dir / "final_model.zip")
    if isinstance(ppo.get_env(), VecNormalize):
        ppo.get_env().save(str(run_dir / "final_vecnormalize.pkl"))
    pd.DataFrame(episodes.rows).to_csv(run_dir / "training_episodes.csv", index=False)
    best = max(selection.log, key=lambda e: (e["valid_breach_rate"], e["mean_return"]))
    meta = {"seed": seed, "threshold": threshold, "timesteps": ppo.num_timesteps, "seconds": seconds,
            "resumed_from": str(resume_from) if resume_from else None, "best": best,
            "selection_episodes": t.selection_episodes, "selection_seed": t.selection_seed}
    (run_dir / "selection_log.json").write_text(json.dumps(selection.log, indent=2))
    (run_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("trained seed %s threshold %s in %.0fs; best %s", seed, threshold, seconds, best)
    return TrainResult(run_dir, seed, threshold, ppo.num_timesteps, seconds, selection.log, best)


def load_policy(run_dir: Path, env: StressTestEnv, which: str = "best", name: str = "ppo") -> SB3Policy:
    """Load ``best`` or ``final`` model with its frozen normalisation statistics."""
    path = run_dir / f"{which}_model.zip"
    model = PPO.load(path, device="cpu")
    stats = vecnormalize_path(path)
    normalizer = None
    if stats.exists():
        normalizer = FrozenObsNormalizer.from_vec_normalize(VecNormalize.load(str(stats), DummyVecEnv([lambda: env])))
    return SB3Policy(model, normalizer, deterministic=True, name=name)
