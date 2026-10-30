"""Stage 6: a library of distinct, plausible, robust stress scenarios built from Stage 5 search output.

Candidates are fixed five-day action paths in Stage 4 action units (one unit = ``max_factor_sigma`` model
standard deviations for every factor when no cap binds, so actions are comparable across factors). This
module only post-processes candidates that the Stage 4 environment has already evaluated: it classifies
their factor-shock paths into analytical families, removes exact and near duplicates, clusters them,
summarises stochastic robustness, attributes the portfolio loss to holdings and assembles a JSON-ready
library. It changes no simulator, reward, limit or plausibility definition.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

FAMILIES = ("broad-market", "technology-led", "financial-led", "energy-led",
            "sector-led (financials and energy jointly)", "mixed-direction")
UNCLASSIFIED = "unclassified"
FACTOR_ORDER = ("SPY_return", "QQQ_return", "XLF_return", "XLE_return")
LIBRARY_SCHEMA_VERSION = "stage6.v2"   # v2: adds the full robustness path (v1 stored executed days only)
SCENARIO_FIELDS = (
    "scenario_id", "status", "selection_reason", "family", "cluster", "discovery_methods", "provenance",
    "representative_candidate_id", "executed_days", "action_path", "factor_shock_path", "robust_action_path",
    "robust_factor_shock_path", "deterministic_rank",
    "deterministic_drawdown", "deterministic_breach_day", "p_valid", "p_valid_ci95", "p_any_breach",
    "p_implausible", "mean_drawdown_at_breach", "mean_breach_day", "screening_p_valid", "screening_episodes",
    "reporting_episodes", "vulnerability",
)


# --- identities ----------------------------------------------------------------------------------

def executed_path(path: np.ndarray, days: int) -> np.ndarray:
    """Actions actually executed (days after a breach are never run)."""
    a = np.asarray(path, dtype=float)
    if a.ndim != 2 or a.shape[1] != len(FACTOR_ORDER):
        raise ValueError(f"path must be (days, {len(FACTOR_ORDER)}), got {a.shape}")
    if not 0 < days <= a.shape[0]:
        raise ValueError("days must be in 1..horizon")
    return a[:days]


def sequence_key(path: np.ndarray, decimals: int = 6) -> str:
    """Exact-duplicate key of a full action path (rounded)."""
    return hashlib.sha1(np.round(np.asarray(path, dtype=float), decimals).tobytes()).hexdigest()


def execution_key(path: np.ndarray, days: int, decimals: int = 6) -> str:
    """Key of the *executed* actions: sequences that differ only after the breach day share it."""
    return hashlib.sha1(np.round(executed_path(path, days), decimals).tobytes()).hexdigest() + f":{days}"


def candidate_id(source: str, path: np.ndarray, seed: Any = None) -> str:
    """Reproducible candidate ID from provenance and the action path."""
    tag = "".join(ch for ch in source.lower() if ch.isalnum())[:12]
    h = hashlib.sha1(f"{source}|{seed}|".encode() + np.round(np.asarray(path, dtype=float), 6).tobytes()).hexdigest()
    return f"{tag}-{h[:10]}"


def factor_shock_path(path: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Daily factor returns ``a * b`` for an action path (factor order as in the Stage 4 simulator)."""
    a = np.asarray(path, dtype=float)
    b = np.asarray(bounds, dtype=float)
    if b.shape != (a.shape[1],):
        raise ValueError("bounds must have one entry per factor")
    return a * b


# --- families ------------------------------------------------------------------------------------

def classify_family(executed: np.ndarray, delta: float = 0.10) -> str:
    """Analytical family of an executed action path (normalised units; factor order SPY, QQQ, XLF, XLE).

    Rules, in order (sums ``s_j`` over executed days, materiality ``delta``): mixed-direction (opposite-sign
    sums beyond delta, or one factor with daily moves of both signs beyond delta); technology-led
    (QQQ < SPY - delta and QQQ most negative); financial-led / energy-led (that sector most negative and below
    the equity mean and the other sector by more than delta); sector-led jointly (both sectors below the equity
    mean by more than delta and within delta of each other); broad-market (equity mean < -delta). Otherwise
    unclassified.
    """
    a = np.asarray(executed, dtype=float)
    s = a.sum(axis=0)
    spy, qqq, xlf, xle = s
    if ((s > delta).any() and (s < -delta).any()) or ((a > delta).any(axis=0) & (a < -delta).any(axis=0)).any():
        return "mixed-direction"
    eq, low = (spy + qqq) / 2, s.min() + 1e-12
    if qqq < spy - delta and qqq <= low:
        return "technology-led"
    if xlf <= low and xlf < eq - delta and xlf < xle - delta:
        return "financial-led"
    if xle <= low and xle < eq - delta and xle < xlf - delta:
        return "energy-led"
    if max(xlf, xle) < eq - delta and abs(xlf - xle) <= delta:
        return "sector-led (financials and energy jointly)"
    if eq < -delta:
        return "broad-market"
    return UNCLASSIFIED


# --- distance and clustering ----------------------------------------------------------------------

def cumulative_feature(path: np.ndarray, days: int) -> np.ndarray:
    """Cumulative normalised shock path, flat after the last executed day (horizon x factors, flattened)."""
    a = np.asarray(path, dtype=float).copy()
    a[days:] = 0.0
    return np.cumsum(a, axis=0).ravel()


def one_step_distance(level: float, horizon: int) -> float:
    """Distance created by one action step ``level`` on one factor on day 1 (it persists for ``horizon`` days)."""
    return float(level * np.sqrt(horizon))


def cluster_features(features: np.ndarray, cut: float) -> np.ndarray:
    """Average-linkage hierarchical clusters cut at Euclidean distance ``cut`` (labels 1..n, deterministic)."""
    X = np.asarray(features, dtype=float)
    if len(X) == 0:
        return np.zeros(0, dtype=int)
    if len(X) == 1:
        return np.ones(1, dtype=int)
    return fcluster(linkage(pdist(X), method="average"), t=cut, criterion="distance").astype(int)


def cluster_sensitivity(features: np.ndarray, cuts: Sequence[float]) -> pd.DataFrame:
    """Number of clusters and of multi-member clusters for each cut distance."""
    rows = []
    for c in cuts:
        lab = cluster_features(features, c)
        cnt = Counter(lab.tolist())
        rows.append({"cut distance": c, "candidates": len(lab), "clusters": len(cnt),
                     "clusters with >= 2 members": sum(v >= 2 for v in cnt.values()),
                     "largest cluster": max(cnt.values()) if cnt else 0})
    return pd.DataFrame(rows)


# --- robustness statistics --------------------------------------------------------------------------

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (NaN for n = 0)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def robustness_summary(records: Sequence[Any]) -> dict[str, Any]:
    """Valid / any / implausible breach probabilities from EpisodeRecord-like objects (denominator: episodes)."""
    n = len(records)
    if n == 0:
        raise ValueError("no episodes")
    kv = sum(bool(r.valid_breach) for r in records)
    br = [r for r in records if r.breached]
    lo, hi = wilson_interval(kv, n)
    valid_ranks = [r.historical_percentile for r in records if r.valid_breach]
    return {"episodes": n, "p_valid": kv / n, "p_valid_ci95": [lo, hi], "p_any_breach": len(br) / n,
            "p_implausible": sum(bool(r.implausible_breach) for r in records) / n,
            "median_valid_rank": float(np.median(valid_ranks)) if valid_ranks else None,
            "mean_drawdown_at_breach": float(np.mean([r.path_drawdown for r in br])) if br else None,
            "mean_breach_day": float(np.mean([r.breach_day for r in br])) if br else None}


# --- portfolio vulnerability ------------------------------------------------------------------------

def holding_contributions(stock_returns: np.ndarray, weights: np.ndarray, capital: float, tickers: Sequence[str]) -> pd.DataFrame:
    """Exact additive P&L decomposition of a constant-mix path.

    ``V_t = V_(t-1) (1 + w'r_t)`` so ``V_T - V_0 = sum_t V_(t-1) sum_i w_i r_(i,t)``: holding i contributes
    ``sum_t V_(t-1) w_i r_(i,t)``. The contributions sum exactly to the total P&L.
    """
    r = np.asarray(stock_returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    values = capital * np.concatenate([[1.0], np.cumprod(1 + r @ w)])
    contrib = (values[:-1, None] * r * w).sum(axis=0)
    total = values[-1] - values[0]
    out = pd.DataFrame({"ticker": list(tickers), "pnl": contrib, "share_of_loss": contrib / total if total != 0 else np.nan})
    out.attrs["total_pnl"], out.attrs["final_value"] = float(total), float(values[-1])
    return out.sort_values("pnl").reset_index(drop=True)


def factor_attribution(factor_shocks: np.ndarray, raw_loadings: np.ndarray, weights: np.ndarray, values: np.ndarray,
                       factor_names: Sequence[str]) -> dict[str, float]:
    """APPROXIMATE P&L attribution to factors via the Stage 4 linear factor model.

    Factor j's share = ``sum_t V_(t-1) (w'B)_j f_(t,j)``; intercept and residual are not attributed (reported as
    remainder by the caller). Model-implied, not causal.
    """
    f = np.asarray(factor_shocks, dtype=float)
    beta_p = np.asarray(weights, dtype=float) @ np.asarray(raw_loadings, dtype=float)
    v_prev = np.asarray(values, dtype=float)[: len(f)]
    return {name: float((v_prev * beta_p[j] * f[:, j]).sum()) for j, name in enumerate(factor_names)}


# --- library ---------------------------------------------------------------------------------------

def scenario_id(representative_candidate: str) -> str:
    """Stable scenario ID derived from its representative candidate."""
    return "S6-" + hashlib.sha1(representative_candidate.encode()).hexdigest()[:8]


def validate_library(library: dict[str, Any], q: float, p_valid_target: float) -> None:
    """Schema and integrity checks for the Stage 6 JSON library (raises ValueError)."""
    for key in ("schema_version", "metadata", "scenarios"):
        if key not in library:
            raise ValueError(f"library lacks '{key}'")
    if library["schema_version"] != LIBRARY_SCHEMA_VERSION:
        raise ValueError("unexpected schema version")
    ids = [s["scenario_id"] for s in library["scenarios"]]
    if len(ids) != len(set(ids)):
        raise ValueError("scenario IDs are not unique")
    for s in library["scenarios"]:
        missing = [f for f in SCENARIO_FIELDS if f not in s]
        if missing:
            raise ValueError(f"{s.get('scenario_id')}: missing fields {missing}")
        if s["status"] not in ("validated", "exploratory"):
            raise ValueError("status must be 'validated' or 'exploratory'")
        if s["status"] == "validated" and not (s["deterministic_rank"] >= q and s["screening_p_valid"] >= p_valid_target):
            raise ValueError(f"{s['scenario_id']}: validated scenario fails the declared selection rule")
        if len(s["action_path"]) != s["executed_days"] or any(len(d) != len(FACTOR_ORDER) for d in s["action_path"]):
            raise ValueError(f"{s['scenario_id']}: action path has wrong shape")
        full = np.asarray(s["robust_action_path"], dtype=float)
        if full.ndim != 2 or full.shape[1] != len(FACTOR_ORDER) or full.shape[0] < s["executed_days"]:
            raise ValueError(f"{s['scenario_id']}: robust_action_path has wrong shape")
        if not np.allclose(full[: s["executed_days"]], np.asarray(s["action_path"], dtype=float)):
            raise ValueError(f"{s['scenario_id']}: robust_action_path does not start with the executed action path")


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    return o


def to_json(obj: Any) -> str:
    """JSON text with numpy types converted and NaN/inf written as null."""
    return json.dumps(_clean(obj), indent=1, allow_nan=False)
