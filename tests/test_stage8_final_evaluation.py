"""Stage 8 final-evaluation helpers: intervals, Kupiec test, overlapping windows, paired bootstrap, nearest scenario."""

from __future__ import annotations

import numpy as np
import pytest

from rlstress.final_evaluation import (
    kupiec_pof,
    merge_overlapping,
    nearest_scenario,
    newcombe_difference,
    non_overlapping,
    paired_bootstrap,
    shortlist_weights,
)


def test_newcombe_difference_contains_estimate_and_is_symmetric() -> None:
    d, lo, hi = newcombe_difference(45, 50, 40, 50)
    assert d == pytest.approx(0.10) and lo < d < hi
    d2, lo2, hi2 = newcombe_difference(40, 50, 45, 50)
    assert d2 == pytest.approx(-d) and lo2 == pytest.approx(-hi) and hi2 == pytest.approx(-lo)
    assert newcombe_difference(50, 50, 50, 50)[1] < 0 < newcombe_difference(50, 50, 50, 50)[2]
    with pytest.raises(ValueError):
        newcombe_difference(1, 0, 1, 1)


def test_kupiec_pof() -> None:
    at = kupiec_pof(25, 500, 0.05)
    assert at["lr_statistic"] == pytest.approx(0.0, abs=1e-9) and at["p_value"] == pytest.approx(1.0)
    hi = kupiec_pof(60, 500, 0.05)
    assert hi["p_value"] < 0.001 and hi["rate"] == pytest.approx(0.12)
    zero = kupiec_pof(0, 500, 0.05)
    assert zero["lr_statistic"] == pytest.approx(-2 * 500 * np.log(0.95)) and zero["p_value"] < 0.001
    with pytest.raises(ValueError):
        kupiec_pof(3, 2, 0.05)


def test_merge_overlapping_and_non_overlapping() -> None:
    assert merge_overlapping([10, 11, 12, 20, 24, 30], 5) == [(10, 12), (20, 24), (30, 30)]
    assert merge_overlapping([], 5) == []
    assert merge_overlapping([3, 8], 5) == [(3, 3), (8, 8)]                      # touching but disjoint windows
    assert list(non_overlapping(12, 5)) == [0, 5, 10] and list(non_overlapping(12, 5, 2)) == [2, 7]
    with pytest.raises(ValueError):
        non_overlapping(10, 5, 5)


def test_shortlist_weights_average_of_shortlist_means() -> None:
    w = shortlist_weights({"one": [[0, 1]], "many": [[0, 1], [2, 3]]}, 4)
    assert np.allclose(w["one"], [0.5, 0.5, 0, 0]) and np.allclose(w["many"], [0.25] * 4)
    with pytest.raises(ValueError):
        shortlist_weights({"bad": [[]]}, 4)


def test_paired_bootstrap_estimates_and_paired_difference() -> None:
    rng = np.random.default_rng(0)
    base = rng.random(400) < 0.9
    y = np.column_stack([base, base, rng.random(400) < 0.5])                       # columns 0 and 1 identical
    r = paired_bootstrap(y, {"a": [[0]], "b": [[1]], "c": [[2]]}, pairs=[("a", "b"), ("a", "c")], n_boot=500, seed=1)
    assert r["a"]["estimate"] == pytest.approx(base.mean()) and r["a"]["ci_low"] <= r["a"]["estimate"] <= r["a"]["ci_high"]
    assert r["a - b"]["ci_low"] == r["a - b"]["ci_high"] == 0.0                      # pairing: identical columns, zero-width
    assert r["a - c"]["ci_low"] > 0
    assert paired_bootstrap(y, {"a": [[0]]}, n_boot=200, seed=3) == paired_bootstrap(y, {"a": [[0]]}, n_boot=200, seed=3)


def test_nearest_scenario() -> None:
    lib = {"x": [0.0, 0.0], "y": [1.0, 1.0], "z": [1.0, 1.0]}
    assert nearest_scenario([0.9, 1.0], lib) == ("y", pytest.approx(0.1))           # tie y/z broken by id
    assert nearest_scenario([0.1, 0.0], lib)[0] == "x"
    with pytest.raises(ValueError):
        nearest_scenario([0.0], {})
