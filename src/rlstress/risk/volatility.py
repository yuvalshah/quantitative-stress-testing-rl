"""Historical volatility.

Estimator: sample standard deviation (``ddof = 1``) of daily simple returns.
Annualised volatility = daily std * sqrt(annualization_days), 252 by default.
Annualised volatility describes dispersion; it is not a loss forecast and is
not converted into VaR here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from rlstress import features as feat
from rlstress.exceptions import InsufficientDataError, MissingDataError

ESTIMATOR = "sample standard deviation (ddof=1) of daily simple returns"


@dataclass(frozen=True)
class VolatilityResult:
    """Volatility of one return series over one window."""

    daily_std: float
    annualized: float
    annualization_days: int
    n_observations: int
    start_date: str
    end_date: str
    estimator: str = ESTIMATOR

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return asdict(self)


def calculate_historical_volatility(
    returns: pd.Series, annualization_days: int = 252, min_observations: int = 2
) -> VolatilityResult:
    """Daily and annualised sample volatility of a complete return series.

    Raises:
        MissingDataError: If ``returns`` has missing values.
        InsufficientDataError: If fewer than ``max(2, min_observations)`` values.
    """
    if returns.isna().any():
        raise MissingDataError("volatility input contains missing returns")
    n = len(returns)
    if n < max(2, min_observations):
        raise InsufficientDataError(f"volatility needs at least {max(2, min_observations)} returns, got {n}")
    daily = float(np.std(returns.to_numpy(dtype=float), ddof=1))
    return VolatilityResult(
        daily_std=daily,
        annualized=daily * float(np.sqrt(annualization_days)),
        annualization_days=annualization_days,
        n_observations=n,
        start_date=returns.index.min().date().isoformat(),
        end_date=returns.index.max().date().isoformat(),
    )


def asset_volatilities(asset_returns: pd.DataFrame, annualization_days: int = 252) -> pd.DataFrame:
    """Per-asset daily and annualised sample volatility."""
    daily = asset_returns.std(ddof=1)
    return pd.DataFrame({"vol_daily": daily, "vol_annualized": daily * np.sqrt(annualization_days)})


def rolling_volatility_table(
    returns: pd.Series, windows: Sequence[int], annualization_days: int = 252, prefix: str = "portfolio"
) -> pd.DataFrame:
    """Trailing ``{prefix}_vol_{w}d`` (daily) and ``{prefix}_vol_ann_{w}d`` (annualised).

    A full window of observations is required (``min_periods = window``).
    """
    out: dict[str, pd.Series] = {}
    for w in windows:
        daily = feat.rolling_volatility(returns, w, w, annualization_days=None)
        out[f"{prefix}_vol_{w}d"] = daily
        out[f"{prefix}_vol_ann_{w}d"] = daily * np.sqrt(annualization_days)
    return pd.DataFrame(out, index=returns.index)
