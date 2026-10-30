"""Stage 7.2: relevance labels from real market weeks and ranking metrics for frozen shortlists.

Relevance is defined from actual returns only, independently of any recommender output:

* **target weeks** of a portfolio in a period: the worst window of every limit-breach episode (overlapping
  breach windows merged) plus the ``n_worst`` worst distinct weeks by realised 5-day drawdown (start gap >= horizon),
  the rule frozen in Stage 8 before its test;
* a scenario **matches** a target week if its cumulative-path feature is within ``tau`` (Stage 6 cluster cut) and its
  family equals the week's family;
* a scenario is **relevant** for that portfolio and period if it matches at least one target week.

Ranking metrics then compare an ordered shortlist with these labels. Search-generated scenarios over-sample extreme
outcomes, so none of these metrics is an unconditional market probability.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from rlstress.final_evaluation import merge_overlapping


def target_weeks(windows: pd.DataFrame, horizon: int, n_worst: int = 5) -> list[tuple[int, str]]:
    """Row positions of the target weeks: one per limit-breach episode, then the ``n_worst`` worst distinct weeks.

    ``windows`` holds one row per rolling window in date order with columns ``breached`` and ``drawdown``.
    """
    w = windows.reset_index(drop=True)
    out: list[tuple[int, str]] = []
    for a, b in merge_overlapping(np.flatnonzero(w["breached"].to_numpy(dtype=bool)), horizon):
        out.append((int(w.loc[a:b, "drawdown"].idxmax()), "limit-breach episode"))
    taken: list[int] = []
    for i in w["drawdown"].sort_values(ascending=False, kind="mergesort").index:
        if all(abs(int(i) - j) >= horizon for j in taken):
            taken.append(int(i))
        if len(taken) == n_worst:
            break
    chosen = {t for t, _ in out}
    out += [(i, "worst distinct week") for i in taken if i not in chosen]
    return out


def match_matrix(target_features: Sequence[Sequence[float]], target_families: Sequence[str],
                 scenario_features: Sequence[Sequence[float]], scenario_families: Sequence[str],
                 tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (targets x scenarios) match matrix and the distance matrix behind it."""
    T = np.atleast_2d(np.asarray(target_features, dtype=float))
    S = np.atleast_2d(np.asarray(scenario_features, dtype=float))
    if len(target_families) != len(T) or len(scenario_families) != len(S):
        raise ValueError("one family per feature row is required")
    d = np.linalg.norm(T[:, None, :] - S[None, :, :], axis=2)
    same = np.asarray(target_families, dtype=object)[:, None] == np.asarray(scenario_families, dtype=object)[None, :]
    return (d <= tau) & same, d


def ranking_metrics(shortlist: Sequence[str], relevant: set[str], n_relevant_in_pool: int, k: int) -> dict[str, float]:
    """Hit@k, Precision@k, MRR and NDCG@k (binary gains) of an ordered shortlist.

    NDCG is NaN when the eligible pool holds no relevant scenario (the ideal ranking is then empty).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    top = list(shortlist)[:k]
    rel = np.array([s in relevant for s in top], dtype=float)
    first = int(np.argmax(rel)) + 1 if rel.any() else 0
    dcg = float(np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))) if len(rel) else 0.0
    n_ideal = min(k, int(n_relevant_in_pool))
    idcg = float(np.sum(1.0 / np.log2(np.arange(2, n_ideal + 2))))
    return {"hit": float(rel.any()), "precision": float(rel.sum() / k), "mrr": 1.0 / first if first else 0.0,
            "ndcg": dcg / idcg if idcg > 0 else float("nan")}


def week_recall(match: np.ndarray, columns: Sequence[int]) -> np.ndarray:
    """Per target week: 1 if any of the scenario ``columns`` matches it (``match`` is targets x scenarios)."""
    m = np.asarray(match, dtype=bool)
    cols = list(columns)
    return m[:, cols].any(axis=1).astype(float) if cols else np.zeros(len(m))


def block_bootstrap_mean(values: Sequence[float], blocks: Sequence[int], n_boot: int = 2000, seed: int = 0,
                         level: float = 0.95) -> dict[str, float]:
    """Mean with a percentile CI from resampling whole time blocks (keeps dependence within a block).

    ``blocks`` gives each value's block label (e.g. the 20-trading-day block of its week's start date); values that
    share a block, such as the same week seen by several portfolios, are resampled together.
    """
    v = np.asarray(values, dtype=float)
    b = np.asarray(blocks)
    if len(v) == 0 or len(v) != len(b):
        raise ValueError("values and blocks must be non-empty and of equal length")
    labels = np.unique(b)
    sums = np.array([v[b == lab].sum() for lab in labels])
    counts = np.array([(b == lab).sum() for lab in labels])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(labels), size=(n_boot, len(labels)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    a = (1 - level) / 2
    return {"estimate": float(v.mean()), "ci_low": float(np.quantile(boot, a)), "ci_high": float(np.quantile(boot, 1 - a)),
            "values": int(len(v)), "blocks": int(len(labels))}


def jaccard_personalisation(lists: Mapping[str, Sequence[str]]) -> float:
    """1 - mean pairwise Jaccard overlap between the shortlists of different portfolios (0 = identical lists)."""
    L = [set(x) for x in lists.values()]
    if len(L) < 2:
        raise ValueError("need at least two shortlists")
    jac = [len(a & c) / len(a | c) if a | c else 1.0 for i, a in enumerate(L) for c in L[i + 1:]]
    return 1.0 - float(np.mean(jac))
