"""Deterministic stress testing with user-specified direct asset-return shocks.

Each shock is the asset's simple return over the whole scenario horizon (for
example 5 trading days); it is applied once, never compounded per day. For a
one-period shock vector ``s``:

    r_p,stress = sum_i w_i * s_i
    V_stress   = V_0 * (1 + r_p,stress)
    loss_pct   = -r_p,stress            (negative = gain)

Shocks below -100% are rejected (a long equity position cannot lose more than
its value); -100% (asset worthless) is allowed. Shocks are hypothetical inputs,
not historical observations or forecasts. Market-factor shocks (SPY, VIX,
rates, oil -> stocks) are not implemented; see docs/stage3_methodology.md.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

import pandas as pd

from rlstress.config import RISK_METRIC_UNITS, RiskLimitSpec
from rlstress.exceptions import InvalidShockError
from rlstress.portfolio import Portfolio, validate_horizon
from rlstress.risk.limits import LimitResult, MetricObservation, evaluate_risk_limits

PATH_METRICS = ("cumulative_loss", "max_drawdown")

REBALANCING_NOTE = (
    "direct shock: one-period return over the horizon applied to target weights; "
    "no intra-horizon rebalancing, costs, taxes or cash flows"
)


@dataclass
class StressResult:
    """Outcome of one deterministic or historical stress scenario."""

    scenario_name: str
    scenario_type: str
    horizon_days: int
    initial_value: float
    final_value: float
    portfolio_return: float
    loss_usd: float
    loss_pct: float
    max_drawdown: float
    asset_contributions: list[dict[str, Any]]
    start_date: str | None = None
    end_date: str | None = None
    value_path: dict[str, float] = field(default_factory=dict)
    limit_results: list[LimitResult] = field(default_factory=list)
    assumptions: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    @property
    def breached(self) -> bool:
        """Whether any evaluated limit was breached."""
        return any(r.breached for r in self.limit_results)

    def evaluate_limits(self, limits: Iterable[RiskLimitSpec], tolerance: float = 1e-12) -> None:
        """Evaluate path limits (cumulative loss, drawdown); 1-day VaR/CVaR limits do not apply to paths."""
        path_limits = [l for l in limits if l.metric in PATH_METRICS]
        self.limit_results = evaluate_risk_limits(path_limits, self.observations(), tolerance)

    def observations(self) -> list[MetricObservation]:
        """Cumulative-loss and drawdown observations over the scenario horizon."""
        common = dict(horizon_days=self.horizon_days, start_date=self.start_date,
                      end_date=self.end_date, source=self.scenario_name)
        return [
            MetricObservation("cumulative_loss", self.loss_pct, RISK_METRIC_UNITS["cumulative_loss"], **common),
            MetricObservation("max_drawdown", -self.max_drawdown, RISK_METRIC_UNITS["max_drawdown"], **common),
        ]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        out = asdict(self)
        out["limit_results"] = [r.to_dict() for r in self.limit_results]
        out["breached"] = self.breached
        return out


def validate_shocks(shocks: Mapping[str, float], tickers: Iterable[str]) -> pd.Series:
    """Shock vector aligned to ``tickers``.

    Raises:
        InvalidShockError: Missing or unknown tickers, non-numeric or non-finite
            values, or shocks below -100%.
    """
    tickers = list(tickers)
    missing = [t for t in tickers if t not in shocks]
    unknown = [t for t in shocks if t not in tickers]
    if missing:
        raise InvalidShockError(f"no shock specified for holdings {missing}")
    if unknown:
        raise InvalidShockError(f"shocks for tickers not in the portfolio: {unknown}")
    for t in tickers:
        s = shocks[t]
        if isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s):
            raise InvalidShockError(f"shock for {t} must be a finite number, got {s!r}")
        if s < -1.0:
            raise InvalidShockError(f"shock for {t} is {s:.4f}; returns below -100% are impossible")
    return pd.Series({t: float(shocks[t]) for t in tickers}, dtype=float)


def apply_asset_return_shocks(
    portfolio: Portfolio,
    shocks: Mapping[str, float],
    horizon_days: int,
    limits: Iterable[RiskLimitSpec] = (),
    name: str = "direct_shock",
    description: str = "",
    tolerance: float = 1e-12,
) -> StressResult:
    """Apply a one-period shock vector to the portfolio and check horizon limits."""
    horizon = validate_horizon(horizon_days)
    s = validate_shocks(shocks, portfolio.tickers)
    w = portfolio.weights
    contrib = w * s
    r_p = float(contrib.sum())
    v0 = portfolio.initial_capital
    v1 = v0 * (1.0 + r_p)
    result = StressResult(
        scenario_name=name,
        scenario_type="direct_asset_shock",
        horizon_days=horizon,
        initial_value=v0,
        final_value=v1,
        portfolio_return=r_p,
        loss_usd=v0 - v1,
        loss_pct=-r_p,
        max_drawdown=min(0.0, r_p),
        asset_contributions=[
            {"ticker": t, "weight": float(w[t]), "shock_return": float(s[t]),
             "contribution_return": float(contrib[t]), "contribution_usd": float(contrib[t] * v0)}
            for t in portfolio.tickers
        ],
        value_path={"start": v0, "end": v1},
        assumptions={"rebalancing": REBALANCING_NOTE, "shock_horizon_days": horizon,
                     "shock_type": "hypothetical direct simple return over the horizon"},
        description=description,
    )
    result.evaluate_limits(limits, tolerance)
    return result
