"""Financial feature engineering.

Stage 2 (implemented): causal rolling statistics on aligned daily data —
volatility, mean return, descriptive drawdown from a rolling high, pairwise
rolling correlations, level statistics and training-fitted regime labels.

Every rolling window is trailing: the value at date ``t`` uses rows ``t-w+1 .. t``
only, and is undefined until ``min_periods`` valid observations exist. Nothing is
back-filled. Windows count rows of the series given (trading days in the
consolidated dataset).

Annualisation: daily standard deviation times ``sqrt(annualization_days)``
(252 by default).

Stage 3 (stubs): the baseline distribution used by the shock simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReturnDistribution:
    """Baseline return distribution used to judge scenario plausibility.

    Attributes:
        tickers: Asset order of ``mean`` and ``cov``.
        mean: Mean daily return vector, shape ``(n_assets,)``.
        cov: Daily return covariance matrix, shape ``(n_assets, n_assets)``.
    """

    tickers: tuple[str, ...]
    mean: np.ndarray
    cov: np.ndarray


def rolling_volatility(
    returns: pd.Series | pd.DataFrame,
    window: int,
    min_periods: int,
    annualization_days: int | None = 252,
) -> pd.Series | pd.DataFrame:
    """Trailing sample standard deviation, annualised by ``sqrt(annualization_days)`` if given."""
    vol = returns.rolling(window, min_periods=min_periods).std(ddof=1)
    return vol * np.sqrt(annualization_days) if annualization_days else vol


def rolling_mean(
    values: pd.Series | pd.DataFrame, window: int, min_periods: int
) -> pd.Series | pd.DataFrame:
    """Trailing mean."""
    return values.rolling(window, min_periods=min_periods).mean()


def rolling_std(
    values: pd.Series | pd.DataFrame, window: int, min_periods: int
) -> pd.Series | pd.DataFrame:
    """Trailing sample standard deviation (not annualised)."""
    return values.rolling(window, min_periods=min_periods).std(ddof=1)


def rolling_drawdown(prices: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Descriptive price drawdown from the trailing ``window``-day high: ``P_t / max(P) - 1``.

    A market-state feature of a single price series, not a portfolio risk metric.
    """
    peak = prices.rolling(window, min_periods=min_periods).max()
    return prices / peak - 1.0


def rolling_correlation(
    x: pd.Series, y: pd.Series, window: int, min_periods: int
) -> pd.Series:
    """Trailing Pearson correlation using only dates where both series are observed."""
    both = x.notna() & y.notna()
    return x.where(both).rolling(window, min_periods=min_periods).corr(y.where(both))


def correlation_matrix(returns: pd.DataFrame, min_periods: int) -> pd.DataFrame:
    """Pairwise-complete Pearson correlation matrix (exploratory)."""
    return returns.corr(min_periods=min_periods)


def fit_quantile_thresholds(values: pd.Series, quantiles: Sequence[float]) -> list[float]:
    """Thresholds at the given quantiles of the non-missing ``values``.

    Call on training-period data only; apply the result to all periods.
    """
    clean = values.dropna()
    if clean.empty:
        raise ValueError("cannot fit thresholds on an empty series")
    return [float(clean.quantile(q)) for q in quantiles]


def assign_regimes(
    values: pd.Series, thresholds: Sequence[float], labels: Sequence[str]
) -> pd.Series:
    """Label each value by fixed thresholds: ``v <= t1`` -> labels[0], ``t1 < v <= t2`` -> labels[1], ...

    Missing values stay missing. Labels depend only on the same-day value and
    the supplied thresholds, so no future information enters.
    """
    if len(labels) != len(thresholds) + 1:
        raise ValueError("need exactly one more label than thresholds")
    bins = [-np.inf, *thresholds, np.inf]
    return pd.cut(values, bins=bins, labels=list(labels), right=True).astype("object").where(
        values.notna()
    )


def return_features(
    returns: pd.DataFrame,
    windows: Sequence[int],
    min_periods: dict[int, int],
    annualization_days: int,
) -> pd.DataFrame:
    """``{A}_vol_ann_{w}d`` and ``{A}_ret_mean_{w}d`` for every ``{A}_return`` column."""
    out: dict[str, pd.Series] = {}
    for col in returns.columns:
        alias = col.removesuffix("_return")
        for w in windows:
            out[f"{alias}_vol_ann_{w}d"] = rolling_volatility(
                returns[col], w, min_periods[w], annualization_days
            )
            out[f"{alias}_ret_mean_{w}d"] = rolling_mean(returns[col], w, min_periods[w])
    return pd.DataFrame(out, index=returns.index)


def drawdown_features(
    prices: pd.DataFrame, windows: Sequence[int], min_periods: dict[int, int]
) -> pd.DataFrame:
    """``{A}_drawdown_{w}d`` for every ``{A}_adj_close`` column."""
    out = {
        f"{col.removesuffix('_adj_close')}_drawdown_{w}d": rolling_drawdown(prices[col], w, min_periods[w])
        for col in prices.columns
        for w in windows
    }
    return pd.DataFrame(out, index=prices.index)


def correlation_features(
    returns: pd.DataFrame,
    pairs: dict[str, Sequence[str]],
    windows: Sequence[int],
    min_periods: dict[int, int],
) -> tuple[pd.DataFrame, list[str]]:
    """``{A}_{B}_corr_{w}d`` for configured (stock, indicator) pairs.

    Returns:
        The features and a list of pairs skipped because a return column is absent.
    """
    out: dict[str, pd.Series] = {}
    skipped: list[str] = []
    for a, indicators in pairs.items():
        for b in indicators:
            ca, cb = f"{a}_return", f"{b}_return"
            if ca not in returns or cb not in returns:
                skipped.append(f"{a}-{b}")
                continue
            for w in windows:
                out[f"{a}_{b}_corr_{w}d"] = rolling_correlation(
                    returns[ca], returns[cb], w, min_periods[w]
                )
    return pd.DataFrame(out, index=returns.index), skipped


def level_features(
    level: pd.Series, prefix: str, windows: Sequence[int], min_periods: dict[int, int]
) -> pd.DataFrame:
    """``{prefix}_mean_{w}d`` and ``{prefix}_std_{w}d`` of a level series (e.g. VIX)."""
    out: dict[str, pd.Series] = {}
    for w in windows:
        out[f"{prefix}_mean_{w}d"] = rolling_mean(level, w, min_periods[w])
        out[f"{prefix}_std_{w}d"] = rolling_std(level, w, min_periods[w])
    return pd.DataFrame(out, index=level.index)


def ewma_covariance(returns: pd.DataFrame, halflife: float = 30.0) -> np.ndarray:
    """Exponentially weighted covariance matrix of the most recent observation (Stage 3)."""
    raise NotImplementedError


def fit_return_distribution(
    returns: pd.DataFrame, window: int | None = None
) -> ReturnDistribution:
    """Estimate the baseline mean and covariance over the last ``window`` rows (Stage 3)."""
    raise NotImplementedError


def build_state_features(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Build per-date market-state observation features for the RL agent (Stage 3+)."""
    raise NotImplementedError
