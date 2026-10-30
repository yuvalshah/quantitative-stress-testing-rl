"""Asset-level risk diagnostics.

Return contribution (fixed weights): ``c_i,t = w_i * r_i,t``; ``sum_i c_i,t = r_p,t``.

Volatility contribution (Euler allocation of portfolio standard deviation),
using the sample covariance ``Sigma`` (ddof = 1) of complete-case daily returns
in the stated window:

    sigma_p = sqrt(w' Sigma w)
    MRC_i   = (Sigma w)_i / sigma_p          (marginal contribution)
    RC_i    = w_i * MRC_i,   sum_i RC_i = sigma_p

These are contributions to volatility only, not to VaR or CVaR, and a larger
weight does not imply a larger contribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from rlstress.exceptions import InsufficientDataError, MissingDataError, ReturnDataError

VOL_CONTRIBUTION_METHOD = (
    "Euler allocation: RC_i = w_i (Sigma w)_i / sqrt(w' Sigma w), Sigma = sample covariance (ddof=1) "
    "of complete-case daily simple returns"
)


def _ordered(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    if set(asset_returns.columns) != set(weights.index):
        raise ReturnDataError("return columns do not match portfolio holdings")
    ordered = asset_returns[list(weights.index)]
    if ordered.isna().any().any():
        raise MissingDataError("contributions need complete returns; apply the missing-data policy first")
    return ordered


def return_contributions(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Per-date ``w_i * r_i,t`` for every holding."""
    ordered = _ordered(asset_returns, weights)
    return ordered * weights.to_numpy(dtype=float)


def contribution_reconciliation_error(contributions: pd.DataFrame, portfolio_returns: pd.Series) -> float:
    """Max ``|sum_i c_i,t - r_p,t|`` across dates."""
    diff = contributions.sum(axis=1) - portfolio_returns.reindex(contributions.index)
    return float(diff.abs().max())


def asset_correlation(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of asset returns over the given (complete-case) window."""
    return asset_returns.corr()


@dataclass
class VolatilityContributions:
    """Euler volatility decomposition of a portfolio."""

    table: pd.DataFrame
    portfolio_vol_daily: float
    portfolio_vol_annualized: float
    n_observations: int
    start_date: str
    end_date: str
    method: str = VOL_CONTRIBUTION_METHOD
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable form (table as records)."""
        return {
            "portfolio_vol_daily": self.portfolio_vol_daily,
            "portfolio_vol_annualized": self.portfolio_vol_annualized,
            "n_observations": self.n_observations,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "method": self.method,
            "by_asset": self.table.reset_index(names="ticker").to_dict(orient="records"),
        }


def volatility_contributions(
    asset_returns: pd.DataFrame, weights: pd.Series, annualization_days: int = 252
) -> VolatilityContributions:
    """Marginal and total contributions to portfolio volatility.

    Raises:
        InsufficientDataError: Fewer than two observations or zero portfolio volatility.
    """
    ordered = _ordered(asset_returns, weights)
    if len(ordered) < 2:
        raise InsufficientDataError("volatility contributions need at least two observations")
    w = weights.to_numpy(dtype=float)
    sigma = np.cov(ordered.to_numpy(dtype=float), rowvar=False, ddof=1)
    sigma_w = sigma @ w
    port_var = float(w @ sigma_w)
    if port_var <= 0:
        raise InsufficientDataError("portfolio volatility is zero; contributions undefined")
    port_vol = float(np.sqrt(port_var))
    mrc = sigma_w / port_vol
    rc = w * mrc
    scale = float(np.sqrt(annualization_days))
    table = pd.DataFrame(
        {
            "weight": w,
            "asset_vol_daily": np.sqrt(np.diag(sigma)),
            "asset_vol_annualized": np.sqrt(np.diag(sigma)) * scale,
            "marginal_contribution_daily": mrc,
            "vol_contribution_daily": rc,
            "vol_contribution_annualized": rc * scale,
            "vol_contribution_share": rc / port_vol,
        },
        index=pd.Index(list(weights.index), name="ticker"),
    )
    return VolatilityContributions(
        table=table, portfolio_vol_daily=port_vol, portfolio_vol_annualized=port_vol * scale,
        n_observations=len(ordered),
        start_date=ordered.index.min().date().isoformat(),
        end_date=ordered.index.max().date().isoformat(),
    )


def extreme_loss_contributions(
    asset_returns: pd.DataFrame, weights: pd.Series, n_dates: int
) -> pd.DataFrame:
    """Return contributions on the ``n_dates`` worst portfolio-return dates (worst first)."""
    contrib = return_contributions(asset_returns, weights)
    port = contrib.sum(axis=1)
    worst = port.nsmallest(n_dates).index
    out = contrib.loc[worst].copy()
    out.insert(0, "portfolio_return", port.loc[worst])
    out.index.name = "date"
    return out


def calculate_asset_risk_contributions(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    annualization_days: int = 252,
    n_extreme_dates: int = 10,
) -> dict[str, Any]:
    """All asset-level diagnostics for one window.

    Returns:
        ``summary`` (per-asset table: weight, volatility, mean return contribution,
        volatility contribution), ``correlation``, ``extreme_losses``,
        ``volatility_contributions`` and the contribution reconciliation error.
    """
    contrib = return_contributions(asset_returns, weights)
    vc = volatility_contributions(asset_returns, weights, annualization_days)
    summary = vc.table.copy()
    summary["mean_return_contribution_daily"] = contrib.mean()
    summary["total_return_contribution_sum"] = contrib.sum()
    port = contrib.sum(axis=1)
    return {
        "summary": summary,
        "correlation": asset_correlation(asset_returns[list(weights.index)]),
        "extreme_losses": extreme_loss_contributions(asset_returns, weights, n_extreme_dates),
        "volatility_contributions": vc,
        "reconciliation_error": contribution_reconciliation_error(contrib, port),
    }
