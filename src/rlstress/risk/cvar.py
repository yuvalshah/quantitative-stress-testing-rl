"""Historical Conditional VaR (Expected Shortfall).

Estimator (Acerbi & Tasche 2002 empirical ES, consistent with the VaR order
statistic in :mod:`rlstress.risk.var`): with returns sorted ascending,
``p = 1 - alpha``, tail mass ``m = n p``, ``j = floor(m)``:

    CVaR_alpha = -( sum_{i=1..j} r_(i) + (m - j) * r_(j+1) ) / m

It is the average loss over exactly the worst ``p`` share of the empirical
distribution: the ``j`` worst observations get full weight and the boundary
observation ``r_(j+1)`` gets the fractional weight ``m - j``. When ``m`` is an
integer no boundary weight is used. Ties need no special handling because only
order statistics enter, so tied values receive their correct total weight.
Because the averaged mass lies at or below ``r_(k)``, ``CVaR >= VaR`` always.

CVaR is an in-sample tail average, not a maximum possible loss.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from rlstress.exceptions import InsufficientDataError
from rlstress.portfolio.validation import validate_capital, validate_confidence
from rlstress.risk.var import sorted_complete_returns, tail_mass, var_order_index

CVAR_ESTIMATOR = (
    "empirical expected shortfall (Acerbi-Tasche): CVaR = -(sum of j worst returns + (m-j) * r_(j+1)) / m, "
    "m = n(1-alpha), j = floor(m)"
)


@dataclass(frozen=True)
class CVaRResult:
    """Historical CVaR of one portfolio-return series."""

    confidence: float
    cvar_pct: float
    cvar_usd: float
    var_pct: float
    capital: float
    n_observations: int
    tail_mass: float
    n_full_tail_observations: int
    boundary_weight: float
    horizon_days: int
    start_date: str
    end_date: str
    window: str
    estimator: str = CVAR_ESTIMATOR
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return asdict(self)


def calculate_historical_cvar(
    returns: pd.Series,
    confidence: float,
    capital: float,
    min_observations: int = 1,
    min_tail_observations: int = 10,
    window: str = "unspecified",
) -> CVaRResult:
    """1-period historical CVaR (fraction and USD) with the VaR threshold it is consistent with."""
    alpha = validate_confidence(confidence)
    cap = validate_capital(capital)
    r = sorted_complete_returns(returns, min_observations)
    n = len(r)
    m = tail_mass(n, alpha)
    j = math.floor(m)
    boundary = round(m - j, 9)
    if m <= 0:  # n(1-alpha) rounds to zero only for alpha within 1e-9 of 1
        raise InsufficientDataError(f"tail mass n(1-alpha) = {n * (1 - alpha):.3g} is effectively zero")
    tail_sum = float(r[:j].sum()) + (boundary * float(r[j]) if boundary > 0 else 0.0)
    cvar = -tail_sum / m
    var = -float(r[var_order_index(n, alpha) - 1])
    warnings: list[str] = []
    if m < min_tail_observations:
        warnings.append(f"tail mass {m:g} observations is below {min_tail_observations}; estimate is unstable")
    if m < 1:
        warnings.append("tail mass below one observation: CVaR equals the single worst loss")
    return CVaRResult(
        confidence=alpha, cvar_pct=cvar, cvar_usd=cvar * cap, var_pct=var, capital=cap,
        n_observations=n, tail_mass=m, n_full_tail_observations=j, boundary_weight=boundary,
        horizon_days=1,
        start_date=returns.index.min().date().isoformat(),
        end_date=returns.index.max().date().isoformat(),
        window=window, warnings=tuple(warnings),
    )
