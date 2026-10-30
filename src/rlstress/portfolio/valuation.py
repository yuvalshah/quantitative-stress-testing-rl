"""Portfolio value paths.

Constant-mix (default): ``V_t = V_(t-1) * (1 + r_p,t)``, ``V_0`` = initial
capital at the close of ``base_date``. Valid for a fully invested portfolio
rebalanced to target weights each period with no costs, taxes, financing or
external cash flows; a real implementation would have to account for those.

Buy-and-hold (separate function): each holding starts at ``w_i * V_0`` and
grows with its own returns, so weights drift with prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rlstress.exceptions import MissingDataError, ReturnDataError
from rlstress.portfolio.validation import validate_capital


def _with_base(index: pd.DatetimeIndex, base_date: pd.Timestamp) -> pd.DatetimeIndex:
    base = pd.Timestamp(base_date)
    if len(index) and base >= index[0]:
        raise ReturnDataError(f"base_date {base.date()} must precede the first return date {index[0].date()}")
    return pd.DatetimeIndex([base]).append(index)


def calculate_portfolio_value(
    portfolio_returns: pd.Series, initial_capital: float, base_date: pd.Timestamp | str
) -> pd.Series:
    """Constant-mix value path including the starting value at ``base_date``.

    Returns:
        Series of length ``len(portfolio_returns) + 1`` named ``portfolio_value``.
    """
    capital = validate_capital(initial_capital)
    if portfolio_returns.isna().any():
        raise MissingDataError("portfolio returns contain missing values")
    growth = np.cumprod(1.0 + portfolio_returns.to_numpy(dtype=float))
    values = np.concatenate([[capital], capital * growth])
    return pd.Series(values, index=_with_base(portfolio_returns.index, base_date), name="portfolio_value")


def buy_and_hold_value(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    initial_capital: float,
    base_date: pd.Timestamp | str,
) -> pd.DataFrame:
    """Buy-and-hold holding values (weights drift) plus a ``total`` column."""
    capital = validate_capital(initial_capital)
    ordered = asset_returns[list(weights.index)]
    if ordered.isna().any().any():
        raise MissingDataError("buy-and-hold valuation needs complete returns")
    growth = np.cumprod(1.0 + ordered.to_numpy(dtype=float), axis=0)
    start = weights.to_numpy(dtype=float) * capital
    holdings = np.vstack([start, growth * start])
    out = pd.DataFrame(holdings, index=_with_base(ordered.index, base_date), columns=list(weights.index))
    out["total"] = out.sum(axis=1)
    return out


def value_consistency_error(values: pd.Series, portfolio_returns: pd.Series) -> float:
    """Max ``|V_t / V_(t-1) - 1 - r_p,t|``; should be ~0 for a consistent path."""
    implied = values.to_numpy()[1:] / values.to_numpy()[:-1] - 1.0
    return float(np.max(np.abs(implied - portfolio_returns.to_numpy()))) if len(implied) else 0.0


def base_date_for(calendar: pd.DatetimeIndex, first_return_date: pd.Timestamp) -> pd.Timestamp:
    """The calendar date immediately before ``first_return_date`` (when capital is invested at the close)."""
    earlier = calendar[calendar < pd.Timestamp(first_return_date)]
    if len(earlier) == 0:
        raise ReturnDataError(f"no calendar date precedes {pd.Timestamp(first_return_date).date()}")
    return earlier[-1]
