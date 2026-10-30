"""Tests for the Stage 7.2 relevance labels and ranking metrics."""

import math

import numpy as np
import pandas as pd
import pytest

from rlstress.unseen_evaluation import (block_bootstrap_mean, jaccard_personalisation, match_matrix, ranking_metrics,
                                        target_weeks, week_recall)


def _windows(drawdowns, breached=None):
    return pd.DataFrame({"drawdown": drawdowns, "breached": breached if breached is not None else [False] * len(drawdowns)})


def test_target_weeks_picks_distinct_worst_weeks():
    dd = [0.01, 0.09, 0.085, 0.02, 0.03, 0.04, 0.08, 0.01, 0.0, 0.05, 0.06, 0.0]
    t = target_weeks(_windows(dd), horizon=5, n_worst=2)
    assert [i for i, _ in t] == [1, 6]                      # 2 is within 5 of 1, so the next distinct week is 6
    assert all(why == "worst distinct week" for _, why in t)


def test_target_weeks_one_per_breach_episode_without_duplicates():
    dd = [0.11, 0.12, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0]
    br = [True, True, True] + [False] * 7
    t = target_weeks(_windows(dd, br), horizon=5, n_worst=2)
    assert t[0] == (1, "limit-breach episode")              # worst window of the merged episode 0..2
    assert [i for i, _ in t] == [1, 8]                      # worst week 1 is not listed twice


def test_match_requires_distance_and_family():
    m, d = match_matrix([[0.0, 0.0], [1.0, 1.0]], ["broad-market", "broad-market"],
                        [[0.1, 0.0], [1.0, 1.1], [0.0, 0.05]], ["broad-market", "broad-market", "technology-led"], tau=0.2)
    assert m.tolist() == [[True, False, False], [False, True, False]]
    assert d.shape == (2, 3) and math.isclose(d[0, 0], 0.1)
    with pytest.raises(ValueError):
        match_matrix([[0.0]], [], [[0.0]], ["x"], 1.0)


def test_ranking_metrics_known_values():
    r = ranking_metrics(["a", "b", "c"], relevant={"b"}, n_relevant_in_pool=2, k=3)
    assert r["hit"] == 1.0 and math.isclose(r["precision"], 1 / 3) and r["mrr"] == 0.5
    assert math.isclose(r["ndcg"], (1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    none = ranking_metrics(["a"], relevant=set(), n_relevant_in_pool=0, k=1)
    assert none["hit"] == 0.0 and none["mrr"] == 0.0 and math.isnan(none["ndcg"])
    perfect = ranking_metrics(["b", "a"], relevant={"b"}, n_relevant_in_pool=1, k=2)
    assert perfect["ndcg"] == 1.0


def test_week_recall():
    m = np.array([[True, False], [False, False], [False, True]])
    assert week_recall(m, [0]).tolist() == [1.0, 0.0, 0.0]
    assert week_recall(m, [0, 1]).tolist() == [1.0, 0.0, 1.0]
    assert week_recall(m, []).tolist() == [0.0, 0.0, 0.0]


def test_block_bootstrap_resamples_blocks():
    out = block_bootstrap_mean([1, 1, 0, 0], blocks=[0, 0, 1, 1], n_boot=500, seed=1)
    assert out["estimate"] == 0.5 and out["blocks"] == 2
    assert out["ci_low"] == 0.0 and out["ci_high"] == 1.0   # only whole blocks are drawn
    const = block_bootstrap_mean([0.3] * 6, blocks=[0, 1, 2, 0, 1, 2], seed=0)
    assert const["ci_low"] == pytest.approx(0.3) and const["ci_high"] == pytest.approx(0.3)


def test_personalisation():
    assert jaccard_personalisation({"p": ["a", "b"], "q": ["a", "b"]}) == 0.0
    assert jaccard_personalisation({"p": ["a"], "q": ["b"]}) == 1.0
    with pytest.raises(ValueError):
        jaccard_personalisation({"p": ["a"]})
