"""Bounded factor-shock actions translated into stock returns.

Action ``a`` in ``[-1, 1]^k`` (one entry per factor) maps to a daily factor return

    f_j = a_j * b_j,   b_j = min(max_factor_sigma * sd_j, max_abs_factor_return)

where ``sd_j`` is the fitted model standard deviation, so ``a = 0`` is a flat
day and ``|a_j| = 1`` is an ``max_factor_sigma``-sigma move (capped). Stock
returns are ``r = a + B f + e`` with the residual ``e`` drawn from the
conditional Student-t of residuals given ``f`` (``residual_mode = "sample"``) or
set to its conditional mean (``"mean"``). Returns ``<= -100%`` or with
``|r| > max_abs_stock_return`` are impossible or out of scope: the residual is
redrawn (rejection sampling, i.e. the conditional distribution truncated to
valid returns, at most ``max_residual_draws`` times); if the factor-implied
mean itself is invalid the step raises :class:`InvalidSimulationError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rlstress.config import SimulationConfig
from rlstress.exceptions import RiskEngineError
from rlstress.shocks.factor_model import ShockModel
from rlstress.shocks.plausibility import Plausibility, factor_plausibility
from rlstress.shocks.tdist import conditional_mvt, sample_mvt

ACTION_TOLERANCE = 1e-9


class InvalidActionError(ValueError):
    """Action has the wrong shape, is not finite, or lies outside [-1, 1]."""


class InvalidSimulationError(RiskEngineError):
    """The simulator could not produce valid stock returns for a factor shock."""


@dataclass
class ShockScenario:
    """A simulated multi-day path of factor shocks and stock returns."""

    tickers: tuple[str, ...]
    factors: tuple[str, ...]
    factor_shocks: np.ndarray       # (n_steps, k)
    stock_returns: np.ndarray       # (n_steps, n)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        """Number of simulated days."""
        return int(self.stock_returns.shape[0])


@dataclass(frozen=True)
class SimulatedStep:
    """One simulated day."""

    action: np.ndarray
    factor_shock: np.ndarray
    expected_returns: np.ndarray
    residuals: np.ndarray
    stock_returns: np.ndarray
    plausibility: Plausibility
    residual_draws: int


class ShockSimulator:
    """Maps actions to factor shocks and stock returns using a fitted :class:`ShockModel`."""

    def __init__(self, model: ShockModel, config: SimulationConfig) -> None:
        self.model = model
        self.config = config
        self.bounds = np.minimum(config.max_factor_sigma * model.factor_sd, config.max_abs_factor_return)
        self._given = np.arange(model.k)

    @property
    def action_dim(self) -> int:
        """Number of factors controlled by the agent."""
        return self.model.k

    def validate_action(self, action: Any) -> np.ndarray:
        """Return the action as a float array, or raise :class:`InvalidActionError`."""
        a = np.asarray(action, dtype=float)
        if a.shape != (self.model.k,):
            raise InvalidActionError(f"action shape {a.shape} != ({self.model.k},)")
        if not np.all(np.isfinite(a)):
            raise InvalidActionError("action contains NaN or infinity")
        if np.any(np.abs(a) > 1.0 + ACTION_TOLERANCE):
            raise InvalidActionError(f"action outside [-1, 1]: {a.tolist()}")
        return np.clip(a, -1.0, 1.0)

    def factor_shock(self, action: Any) -> np.ndarray:
        """Daily factor returns for a valid action."""
        return self.validate_action(action) * self.bounds

    def _valid(self, r: np.ndarray) -> bool:
        return bool(np.all(r > -1.0) and np.all(np.abs(r) <= self.config.max_abs_stock_return))

    def stock_returns(
        self, f: np.ndarray, rng: np.random.Generator, mode: str | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Stock returns for factor shock ``f``: ``(expected, residual, returns, draws)``."""
        mode = mode or self.config.residual_mode
        expected = self.model.expected_returns(f)
        mu_c, sigma_c, nu_c = conditional_mvt(self.model.mu, self.model.sigma, self.model.nu, self._given, f)
        if not self._valid(expected + mu_c):
            raise InvalidSimulationError(
                f"factor-implied returns are invalid (min {float((expected + mu_c).min()):.3f}, "
                f"max {float((expected + mu_c).max()):.3f})")
        if mode == "mean":
            return expected, mu_c, expected + mu_c, 0
        for draw in range(1, self.config.max_residual_draws + 1):
            e = sample_mvt(rng, mu_c, sigma_c, nu_c, 1)[0]
            r = expected + e
            if self._valid(r):
                return expected, e, r, draw
        raise InvalidSimulationError(
            f"no valid residual draw in {self.config.max_residual_draws} attempts")

    def step(self, action: Any, rng: np.random.Generator, mode: str | None = None) -> SimulatedStep:
        """Simulate one day for an agent action."""
        a = self.validate_action(action)
        f = a * self.bounds
        expected, e, r, draws = self.stock_returns(f, rng, mode)
        return SimulatedStep(a, f, expected, e, r, factor_plausibility(self.model, f), draws)

    def sample_days(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        """``n`` unconditional model days: (factor returns, stock returns), from the joint t."""
        z = sample_mvt(rng, self.model.mu, self.model.sigma, self.model.nu, n)
        f, e = z[:, : self.model.k], z[:, self.model.k:]
        return f, self.model.raw_intercept + f @ self.model.raw_loadings.T + e
