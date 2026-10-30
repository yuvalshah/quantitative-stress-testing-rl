"""Policies evaluated under one protocol: zero, scripted, random baselines and trained PPO.

Every policy exposes ``reset(seed)`` (called at the start of each evaluation
episode with the episode seed) and ``act(obs)`` returning an action in
``[-1, 1]^k``.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class Policy(Protocol):
    """Interface used by :func:`rlstress.rl.evaluation.run_episode`."""

    name: str

    def reset(self, seed: int) -> None:
        """Prepare for an episode."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Action for an observation."""


class ConstantPolicy:
    """The same action every step (zero baseline, scripted decline)."""

    def __init__(self, action: np.ndarray, name: str) -> None:
        self.action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.name = name

    def reset(self, seed: int) -> None:
        """Stateless."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Fixed action."""
        return self.action.copy()


def zero_policy(k: int) -> ConstantPolicy:
    """No deliberate stress: flat factor days."""
    return ConstantPolicy(np.zeros(k), "zero_action")


def scripted_decline_policy(k: int, level: float) -> ConstantPolicy:
    """Stage 4 steady, market-wide moderate decline: ``level`` on every factor every day."""
    return ConstantPolicy(np.full(k, level), "scripted_moderate_decline")


class RandomPolicy:
    """Uniform actions on ``[-1, 1]^k`` from an RNG re-seeded per episode (``seed + offset``)."""

    def __init__(self, k: int, offset: int, name: str = "random_action") -> None:
        self.k = k
        self.offset = offset
        self.name = name
        self.rng = np.random.default_rng(offset)

    def reset(self, seed: int) -> None:
        """Seed the action RNG from the episode seed."""
        self.rng = np.random.default_rng(seed + self.offset)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Uniform random action."""
        return self.rng.uniform(-1.0, 1.0, self.k).astype(np.float32)


class FrozenObsNormalizer:
    """Observation normalisation with fixed statistics (SB3 ``VecNormalize`` formula, no updates)."""

    def __init__(self, mean: np.ndarray, var: np.ndarray, clip_obs: float, epsilon: float) -> None:
        self.mean = np.asarray(mean, dtype=np.float64).copy()
        self.var = np.asarray(var, dtype=np.float64).copy()
        self.clip_obs = float(clip_obs)
        self.epsilon = float(epsilon)

    @classmethod
    def from_vec_normalize(cls, vn: Any) -> FrozenObsNormalizer:
        """Snapshot the current statistics of a ``VecNormalize``."""
        return cls(vn.obs_rms.mean, vn.obs_rms.var, vn.clip_obs, vn.epsilon)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """``clip((obs - mean) / sqrt(var + eps), -clip, clip)``."""
        out = (np.asarray(obs, dtype=np.float64) - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(out, -self.clip_obs, self.clip_obs).astype(np.float32)


class SB3Policy:
    """A trained stable-baselines3 model with frozen observation normalisation.

    ``obs_normalizer`` is a :class:`FrozenObsNormalizer` (or ``None``); its
    statistics never change during evaluation. Actions are clipped to the Box
    exactly as SB3 does during training.
    """

    def __init__(self, model: Any, obs_normalizer: Any = None, deterministic: bool = True,
                 name: str = "ppo") -> None:
        self.model = model
        self.normalizer = obs_normalizer
        self.deterministic = deterministic
        self.name = name

    def reset(self, seed: int) -> None:
        """Deterministic policies need no per-episode state."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Policy mean action (deterministic) clipped to [-1, 1]."""
        x = obs if self.normalizer is None else self.normalizer.normalize_obs(obs)
        action, _ = self.model.predict(x, deterministic=self.deterministic)
        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
