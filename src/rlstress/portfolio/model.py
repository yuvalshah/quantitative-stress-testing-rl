"""The :class:`Portfolio` value object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from rlstress.config import PortfolioConfig
from rlstress.portfolio.validation import (
    DEFAULT_WEIGHT_TOLERANCE,
    validate_capital,
    validate_weights,
)


@dataclass(frozen=True, eq=False)
class Portfolio:
    """A validated, fully invested portfolio with target weights.

    Attributes:
        name: Identifier used in results.
        weights: Target weights indexed by ticker (decimal fractions summing to 1).
        initial_capital: Starting value in ``currency``.
        currency: Base currency.
        policy: Weight policy (``"long_only"``).
        rebalancing: ``"daily_constant_mix"``: weights reset to target each period.
        weight_tolerance: Allowed ``|sum(weights) - 1|``.
    """

    name: str
    weights: pd.Series
    initial_capital: float
    currency: str = "USD"
    policy: str = "long_only"
    rebalancing: str = "daily_constant_mix"
    weight_tolerance: float = DEFAULT_WEIGHT_TOLERANCE
    universe: tuple[str, ...] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        validated = validate_weights(
            self.weights.to_dict() if isinstance(self.weights, pd.Series) else dict(self.weights),
            self.universe, self.policy, self.weight_tolerance,
        )
        object.__setattr__(self, "weights", validated)
        object.__setattr__(self, "initial_capital", validate_capital(self.initial_capital))

    @classmethod
    def from_config(cls, config: PortfolioConfig, universe: Iterable[str] | None = None) -> Portfolio:
        """Build and validate a portfolio from the ``[portfolio]`` config section."""
        return cls(
            name=config.name,
            weights=pd.Series(dict(zip(config.tickers, config.weights)), dtype=float),
            initial_capital=config.initial_capital,
            currency=config.base_currency,
            policy=config.policy,
            rebalancing=config.rebalancing,
            weight_tolerance=config.weight_tolerance,
            universe=tuple(universe) if universe is not None else None,
        )

    @property
    def tickers(self) -> tuple[str, ...]:
        """Holdings in weight order."""
        return tuple(self.weights.index)

    @property
    def n_assets(self) -> int:
        """Number of holdings."""
        return len(self.weights)

    def weight_vector(self) -> np.ndarray:
        """Weights as an array in :attr:`tickers` order."""
        return self.weights.to_numpy(dtype=float)

    def exposures(self) -> pd.Series:
        """Initial currency exposure per holding (``weight * initial_capital``)."""
        return self.weights * self.initial_capital

    def with_weights(self, new_weights: Mapping[str, float]) -> Portfolio:
        """A copy with new (validated) weights."""
        return Portfolio(
            self.name, pd.Series(dict(new_weights), dtype=float), self.initial_capital,
            self.currency, self.policy, self.rebalancing, self.weight_tolerance, self.universe,
        )

    def describe(self) -> dict[str, Any]:
        """JSON-serialisable configuration summary."""
        return {
            "name": self.name,
            "weights": {k: float(v) for k, v in self.weights.items()},
            "initial_capital": self.initial_capital,
            "currency": self.currency,
            "policy": self.policy,
            "rebalancing": self.rebalancing,
            "weight_tolerance": self.weight_tolerance,
        }
