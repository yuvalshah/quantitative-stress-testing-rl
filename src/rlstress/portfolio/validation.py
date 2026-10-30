"""Reusable validation of portfolio weights, capital, confidence levels, horizons and return data.

Nothing here normalises or repairs input; invalid input raises. Normalisation is
a separate, explicit operation (:func:`normalize_weights`).

Only the ``long_only`` policy (fully invested, unleveraged: ``w_i >= 0`` and
``sum(w) = 1`` within tolerance) is supported. Other policies are rejected
rather than validated with long-only rules, so short or levered portfolios can
be added later with their own checks.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Iterable, Mapping

import pandas as pd

from rlstress.exceptions import PortfolioValidationError, ReturnDataError

SUPPORTED_POLICIES = ("long_only",)
DEFAULT_WEIGHT_TOLERANCE = 1e-8


def _is_number(x: object) -> bool:
    return isinstance(x, Real) and not isinstance(x, bool)


def validate_weights(
    weights: Mapping[str, float],
    universe: Iterable[str] | None = None,
    policy: str = "long_only",
    tolerance: float = DEFAULT_WEIGHT_TOLERANCE,
) -> pd.Series:
    """Validate a ticker-to-weight mapping and return it as a float Series.

    Args:
        weights: Ticker to weight (decimal fraction, 0.125 = 12.5%).
        universe: Allowed tickers; ``None`` skips the membership check.
        policy: Weight policy; only ``"long_only"`` is implemented.
        tolerance: Allowed ``|sum(weights) - 1|``.

    Raises:
        PortfolioValidationError: Empty mapping, unknown ticker, non-numeric or
            non-finite weight, negative weight, wrong sum, or unsupported policy.
    """
    if policy not in SUPPORTED_POLICIES:
        raise PortfolioValidationError(
            f"weight policy {policy!r} is not implemented; supported: {SUPPORTED_POLICIES}"
        )
    if not weights:
        raise PortfolioValidationError("portfolio has no holdings")
    if universe is not None:
        allowed = set(universe)
        unknown = [t for t in weights if t not in allowed]
        if unknown:
            raise PortfolioValidationError(f"tickers not in the asset universe: {unknown}")
    for ticker, w in weights.items():
        if not _is_number(w):
            raise PortfolioValidationError(f"weight for {ticker} is not a number: {w!r}")
        if not math.isfinite(float(w)):
            raise PortfolioValidationError(f"weight for {ticker} is not finite: {w!r}")
        if w < 0:
            raise PortfolioValidationError(f"negative weight for {ticker} not allowed under long_only")
    total = math.fsum(float(w) for w in weights.values())
    if abs(total - 1.0) > tolerance:
        raise PortfolioValidationError(
            f"weights sum to {total!r}; long_only requires 1 within {tolerance}. "
            "Use normalize_weights() explicitly if rescaling is intended."
        )
    return pd.Series({t: float(w) for t, w in weights.items()}, dtype=float)


def normalize_weights(weights: Mapping[str, float]) -> pd.Series:
    """Explicitly rescale finite, non-negative weights so they sum to one."""
    values = pd.Series(dict(weights), dtype=float)
    if values.empty or not values.map(math.isfinite).all() or (values < 0).any():
        raise PortfolioValidationError("can only normalise finite, non-negative weights")
    total = values.sum()
    if total <= 0:
        raise PortfolioValidationError("cannot normalise weights summing to zero")
    return values / total


def validate_capital(capital: float) -> float:
    """Positive, finite starting capital."""
    if not _is_number(capital) or not math.isfinite(float(capital)) or capital <= 0:
        raise PortfolioValidationError(f"capital must be a positive finite number, got {capital!r}")
    return float(capital)


def validate_confidence(alpha: float) -> float:
    """Confidence level with ``0 < alpha < 1``."""
    if not _is_number(alpha) or not 0 < float(alpha) < 1:
        raise PortfolioValidationError(f"confidence level must satisfy 0 < alpha < 1, got {alpha!r}")
    return float(alpha)


def validate_horizon(days: int) -> int:
    """Positive integer horizon in trading days."""
    if not isinstance(days, int) or isinstance(days, bool) or days < 1:
        raise PortfolioValidationError(f"horizon must be a positive integer, got {days!r}")
    return days


def validate_return_frame(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> None:
    """Check a return frame: DatetimeIndex, sorted, unique dates, required columns, no returns < -100%.

    Raises:
        ReturnDataError: Describing the first problem found.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ReturnDataError("returns must be indexed by a DatetimeIndex")
    if frame.empty:
        raise ReturnDataError("return data are empty")
    if not frame.index.is_unique:
        dup = frame.index[frame.index.duplicated()].unique()[:5]
        raise ReturnDataError(f"duplicate dates in returns: {[d.date().isoformat() for d in dup]}")
    if not frame.index.is_monotonic_increasing:
        raise ReturnDataError("return dates are not sorted ascending")
    if columns is not None:
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ReturnDataError(f"return data lack columns {missing}")
    checked = frame[list(columns)] if columns is not None else frame
    numeric = checked.select_dtypes("number")
    bad = (numeric < -1).any()
    if bad.any():
        raise ReturnDataError(
            f"returns below -100% are impossible for long equity positions: {list(bad[bad].index)}"
        )


def validate_portfolio(
    weights: Mapping[str, float],
    capital: float,
    universe: Iterable[str] | None = None,
    policy: str = "long_only",
    tolerance: float = DEFAULT_WEIGHT_TOLERANCE,
) -> tuple[pd.Series, float]:
    """Validate weights and capital together; returns ``(weights, capital)``."""
    return validate_weights(weights, universe, policy, tolerance), validate_capital(capital)
