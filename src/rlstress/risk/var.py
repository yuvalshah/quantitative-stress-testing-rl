"""Historical-simulation Value at Risk.

Sign convention: losses are positive, ``L_t = -r_t``.

Estimator (inverted empirical CDF, numpy ``method="inverted_cdf"``): sort the
``n`` portfolio returns ascending, ``r_(1) <= ... <= r_(n)``, let
``p = 1 - alpha`` and ``k = ceil(n * p)``. Then

    VaR_alpha = -r_(k)

i.e. the loss that is exceeded with empirical probability at most ``p``. A
negative VaR (the ``k``-th worst day was a gain) is reported as is, never
clipped. Portfolio VaR always comes from the joint portfolio-return series,
never from per-asset quantiles.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from rlstress.exceptions import InsufficientDataError, MissingDataError
from rlstress.portfolio.validation import validate_capital, validate_confidence

VAR_ESTIMATOR = "historical simulation: VaR = -r_(k), k = ceil(n(1-alpha)), returns sorted ascending (inverted empirical CDF)"
_ROUND = 9  # guards ceil/floor against binary noise, e.g. 100 * (1 - 0.95) = 5.000000000000004


def tail_mass(n: int, alpha: float) -> float:
    """``n * (1 - alpha)`` rounded to remove floating-point noise."""
    return round(n * (1.0 - alpha), _ROUND)


def var_order_index(n: int, alpha: float) -> int:
    """1-based order-statistic index ``k = ceil(n (1 - alpha))``, at least 1."""
    return max(1, math.ceil(tail_mass(n, alpha)))


def sorted_complete_returns(returns: pd.Series, min_observations: int) -> np.ndarray:
    """Ascending returns after checking for missing values and sample size."""
    if returns.isna().any():
        raise MissingDataError("VaR/CVaR input contains missing returns")
    n = len(returns)
    if n == 0:
        raise InsufficientDataError("VaR/CVaR need at least one return")
    if n < min_observations:
        raise InsufficientDataError(f"VaR/CVaR need at least {min_observations} returns, got {n}")
    return np.sort(returns.to_numpy(dtype=float))


@dataclass(frozen=True)
class VaRResult:
    """Historical VaR of one portfolio-return series."""

    confidence: float
    var_pct: float
    var_usd: float
    capital: float
    n_observations: int
    order_index: int
    horizon_days: int
    start_date: str
    end_date: str
    window: str
    estimator: str = VAR_ESTIMATOR
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return asdict(self)


def calculate_historical_var(
    returns: pd.Series,
    confidence: float,
    capital: float,
    min_observations: int = 1,
    min_tail_observations: int = 10,
    window: str = "unspecified",
) -> VaRResult:
    """1-period historical VaR (fraction and USD) of a portfolio-return series.

    Args:
        returns: Complete portfolio returns over the estimation window.
        confidence: ``alpha`` with ``0 < alpha < 1``.
        capital: Portfolio value the percentage is applied to.
        min_observations: Refuse estimation below this sample size.
        min_tail_observations: Add a warning when ``n(1-alpha)`` is below this.
        window: Label of the estimation window (recorded in the result).
    """
    alpha = validate_confidence(confidence)
    cap = validate_capital(capital)
    r = sorted_complete_returns(returns, min_observations)
    n = len(r)
    k = var_order_index(n, alpha)
    var = -float(r[k - 1])
    warnings: list[str] = []
    if tail_mass(n, alpha) < min_tail_observations:
        warnings.append(f"only {tail_mass(n, alpha):g} expected tail observations; estimate is unstable")
    if var < 0:
        warnings.append("negative VaR: the quantile return is a gain; no loss at this confidence in-sample")
    return VaRResult(
        confidence=alpha, var_pct=var, var_usd=var * cap, capital=cap, n_observations=n,
        order_index=k, horizon_days=1,
        start_date=returns.index.min().date().isoformat(),
        end_date=returns.index.max().date().isoformat(),
        window=window, warnings=tuple(warnings),
    )
