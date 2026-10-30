"""Validation of the fitted shock model against empirical training-period behaviour."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from rlstress.portfolio import select_window
from rlstress.risk import calculate_historical_cvar, calculate_historical_var
from rlstress.shocks.factor_model import ShockModel, _ols
from rlstress.shocks.simulator import ShockSimulator
from rlstress.shocks.tdist import mahalanobis_sq

QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.999)
TAIL_LEVELS = (0.05, 0.01, 0.001)


def mahalanobis_calibration(
    model: ShockModel, factor_returns: pd.DataFrame, which: str = "plausibility"
) -> dict[str, Any]:
    """Empirical vs model distribution of ``delta / k`` on training factor days.

    Under the model ``delta / k ~ F(k, nu)``. Compares quantiles and the share of
    training days whose tail probability falls below 5%, 1% and 0.1%.
    ``which`` = ``"plausibility"`` (factor-only t, used for scoring) or
    ``"joint"`` (factor marginal of the joint t, used for simulation).
    """
    f = factor_returns[list(model.factors)].to_numpy(dtype=float)
    if which == "plausibility":
        mu, sigma, nu = model.plaus_mu, model.plaus_sigma, model.plaus_nu
    else:
        mu, sigma, nu = model.factor_mu, model.factor_sigma, model.nu
    stat = mahalanobis_sq(f, mu, sigma) / model.k
    p = stats.f.sf(stat, model.k, nu)
    return {
        "model": which, "nu": nu,
        "n_days": int(len(stat)),
        "quantiles": [
            {"q": q, "empirical": float(np.quantile(stat, q)), "model": float(stats.f.ppf(q, model.k, nu))}
            for q in QUANTILES
        ],
        "tail_rates": [{"level": a, "empirical_share": float((p < a).mean()), "nominal": a} for a in TAIL_LEVELS],
    }


def simulated_vs_empirical(
    simulator: ShockSimulator,
    stock_returns: pd.DataFrame,
    weights: pd.Series,
    capital: float,
    n: int,
    seed: int,
    confidence_levels: Sequence[float] = (0.95, 0.99),
) -> dict[str, Any]:
    """Compare model-simulated days with training days: per-stock tails and portfolio VaR/CVaR.

    Portfolio VaR/CVaR on both samples use the Stage 3 historical estimators.
    """
    rng = np.random.default_rng(seed)
    _, sim = simulator.sample_days(rng, n)
    tickers = list(simulator.model.tickers)
    emp = stock_returns[tickers].to_numpy(dtype=float)
    per_stock = []
    for i, t in enumerate(tickers):
        per_stock.append({
            "ticker": t,
            "sd_empirical": float(emp[:, i].std(ddof=1)), "sd_model": float(sim[:, i].std(ddof=1)),
            "q01_empirical": float(np.quantile(emp[:, i], 0.01)), "q01_model": float(np.quantile(sim[:, i], 0.01)),
            "q99_empirical": float(np.quantile(emp[:, i], 0.99)), "q99_model": float(np.quantile(sim[:, i], 0.99)),
        })
    w = weights[tickers].to_numpy(dtype=float)
    idx_emp = stock_returns.index
    port_emp = pd.Series(emp @ w, index=idx_emp)
    port_sim = pd.Series(sim @ w, index=pd.bdate_range("2000-01-03", periods=n))  # labels only
    port = []
    for a in confidence_levels:
        port.append({
            "confidence": a,
            "var_empirical": calculate_historical_var(port_emp, a, capital).var_pct,
            "var_model": calculate_historical_var(port_sim, a, capital).var_pct,
            "cvar_empirical": calculate_historical_cvar(port_emp, a, capital).cvar_pct,
            "cvar_model": calculate_historical_cvar(port_sim, a, capital).cvar_pct,
        })
    return {"n_simulated_days": n, "seed": seed, "per_stock": per_stock, "portfolio": port,
            "portfolio_sd_empirical": float(port_emp.std(ddof=1)),
            "portfolio_sd_model": float(port_sim.std(ddof=1))}


def candidate_factor_diagnostics(
    model: ShockModel,
    source: pd.DataFrame,
    candidates: Sequence[str],
    start: str,
    end: str,
    return_template: str = "{ticker}_return",
) -> list[dict[str, Any]]:
    """Incremental explanatory power of each candidate factor over the fitted model.

    For each candidate: average increase in stock-regression R^2 when it is added
    to the model factors (complete cases in the training window), and its
    correlation with the market factor. Used to decide whether to include it.
    """
    out = []
    stock_cols = [return_template.format(ticker=t) for t in model.tickers]
    for cand in candidates:
        if cand not in source.columns:
            out.append({"factor": cand, "status": "column not available"})
            continue
        win = select_window(source[[*model.factors, cand, *stock_cols]], start, end).dropna()
        base_x = win[list(model.factors)].to_numpy()
        full_x = win[[*model.factors, cand]].to_numpy()
        gains = []
        for col in stock_cols:
            y = win[col].to_numpy()
            _, r0 = _ols(y, base_x)
            _, r1 = _ols(y, full_x)
            gains.append((r0.var() - r1.var()) / y.var())
        out.append({
            "factor": cand, "n_obs": int(len(win)),
            "mean_incremental_r2": float(np.mean(gains)), "max_incremental_r2": float(np.max(gains)),
            "corr_with_market": float(np.corrcoef(win[model.market_factor], win[cand])[0, 1]),
        })
    return out
