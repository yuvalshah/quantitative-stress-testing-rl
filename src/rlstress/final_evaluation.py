"""Stage 8 final out-of-sample evaluation: small statistical helpers (no model fitting, no data access).

Everything here is pure: proportions and their differences (Wilson / Newcombe), the Kupiec
proportion-of-failures test for VaR exceedances, handling of overlapping rolling windows
(merging into distinct episodes, non-overlapping subsamples), an episode-level paired bootstrap
for shortlist averages under common random numbers, and nearest-scenario lookup in the
Stage 6 cumulative-path feature space.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy import stats

from rlstress.scenario_library import wilson_interval


def newcombe_difference(k1: int, n1: int, k2: int, n2: int, z: float = 1.96) -> tuple[float, float, float]:
    """``p1 - p2`` with Newcombe's hybrid-score 95% interval (method 10; independent samples)."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("both samples need at least one observation")
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_interval(k1, n1, z)
    l2, u2 = wilson_interval(k2, n2, z)
    d = p1 - p2
    lo = d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return float(d), float(max(-1.0, lo)), float(min(1.0, hi))


def kupiec_pof(exceedances: int, n: int, p: float) -> dict[str, float]:
    """Kupiec (1995) unconditional-coverage likelihood-ratio test of ``exceedances`` in ``n`` days at rate ``p``."""
    if not 0 < p < 1 or n <= 0 or not 0 <= exceedances <= n:
        raise ValueError("need 0 < p < 1, n > 0 and 0 <= exceedances <= n")
    x = exceedances
    phat = x / n

    def ll(q: float) -> float:
        return (n - x) * np.log1p(-q) if x == 0 else (x * np.log(q) if x == n else (n - x) * np.log1p(-q) + x * np.log(q))

    lr = float(-2.0 * (ll(p) - ll(phat))) if 0 < phat < 1 else float(-2.0 * (ll(p) - 0.0))
    return {"exceedances": int(x), "n": int(n), "rate": float(phat), "expected_rate": float(p), "expected": float(n * p),
            "lr_statistic": max(0.0, lr), "p_value": float(stats.chi2.sf(max(0.0, lr), 1))}


def merge_overlapping(positions: Sequence[int], horizon: int) -> list[tuple[int, int]]:
    """Group window start positions whose windows overlap (start gap < ``horizon``) into episodes ``(first, last)``."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    groups: list[tuple[int, int]] = []
    for p in sorted(int(x) for x in positions):
        if groups and p - groups[-1][1] < horizon:
            groups[-1] = (groups[-1][0], p)
        else:
            groups.append((p, p))
    return groups


def non_overlapping(n_windows: int, horizon: int, offset: int = 0) -> np.ndarray:
    """Start positions ``offset, offset + horizon, ...`` of disjoint windows among ``n_windows`` rolling windows."""
    if horizon < 1 or not 0 <= offset < horizon:
        raise ValueError("need horizon >= 1 and 0 <= offset < horizon")
    return np.arange(offset, n_windows, horizon)


def shortlist_weights(groups: Mapping[str, Sequence[Sequence[int]]], n_columns: int) -> dict[str, np.ndarray]:
    """Linear weights turning per-scenario means into each group's statistic.

    A group is one or more shortlists (column indices); its statistic is the average over shortlists of the
    shortlist mean, so a random baseline of many shortlists is a single weight vector.
    """
    out = {}
    for name, lists in groups.items():
        w = np.zeros(n_columns)
        if not lists or any(len(ids) == 0 for ids in lists):
            raise ValueError(f"group {name!r} has an empty shortlist")
        for ids in lists:
            w[list(ids)] += 1.0 / (len(ids) * len(lists))
        out[name] = w
    return out


def paired_bootstrap(outcomes: np.ndarray, groups: Mapping[str, Sequence[Sequence[int]]],
                     pairs: Sequence[tuple[str, str]] = (), n_boot: int = 2000, seed: int = 0,
                     level: float = 0.95) -> dict[str, dict[str, float]]:
    """Percentile CIs of group averages and paired differences, resampling *episodes* (rows) jointly.

    ``outcomes`` is (episodes x scenarios), every scenario evaluated on the same episode seeds (common random
    numbers), so resampling rows keeps the pairing between scenarios and between groups.
    """
    y = np.asarray(outcomes, dtype=float)
    if y.ndim != 2 or len(y) == 0:
        raise ValueError("outcomes must be a non-empty episodes x scenarios matrix")
    weights = shortlist_weights(groups, y.shape[1])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    col_means = np.stack([y[i].mean(axis=0) for i in idx])          # n_boot x scenarios
    point = y.mean(axis=0)
    a = (1 - level) / 2
    res: dict[str, dict[str, float]] = {}
    for name, w in weights.items():
        b = col_means @ w
        res[name] = {"estimate": float(point @ w), "ci_low": float(np.quantile(b, a)), "ci_high": float(np.quantile(b, 1 - a))}
    for g1, g2 in pairs:
        b = col_means @ (weights[g1] - weights[g2])
        res[f"{g1} - {g2}"] = {"estimate": float(point @ (weights[g1] - weights[g2])),
                               "ci_low": float(np.quantile(b, a)), "ci_high": float(np.quantile(b, 1 - a))}
    return res


def nearest_scenario(feature: Sequence[float], library: Mapping[str, Sequence[float]]) -> tuple[str, float]:
    """Library scenario with the smallest Euclidean distance in the cumulative-path feature space (ties: id order)."""
    if not library:
        raise ValueError("empty library")
    f = np.asarray(feature, dtype=float)
    best = min(sorted(library), key=lambda sid: float(np.linalg.norm(f - np.asarray(library[sid], dtype=float))))
    return best, float(np.linalg.norm(f - np.asarray(library[best], dtype=float)))
