"""Stage 6 scenario library: identities, families, duplicates, clustering, robustness, attribution, schema."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from rlstress.scenario_library import (
    FACTOR_ORDER,
    LIBRARY_SCHEMA_VERSION,
    SCENARIO_FIELDS,
    candidate_id,
    classify_family,
    cluster_features,
    cluster_sensitivity,
    cumulative_feature,
    execution_key,
    executed_path,
    factor_attribution,
    factor_shock_path,
    holding_contributions,
    one_step_distance,
    robustness_summary,
    scenario_id,
    sequence_key,
    to_json,
    validate_library,
    wilson_interval,
)


def test_factor_order_and_shock_dimensions() -> None:
    assert FACTOR_ORDER == ("SPY_return", "QQQ_return", "XLF_return", "XLE_return")
    path = np.full((5, 4), -0.15)
    shocks = factor_shock_path(path, np.array([0.1267, 0.1596, 0.1692, 0.2336]))
    assert shocks.shape == (5, 4) and shocks[0, 0] == pytest.approx(-0.019005)
    with pytest.raises(ValueError):
        factor_shock_path(path, np.ones(3))
    with pytest.raises(ValueError):
        executed_path(np.zeros((5, 3)), 2)
    assert executed_path(path, 3).shape == (3, 4)


def test_ids_reproducible_and_provenance_sensitive() -> None:
    p = np.full((5, 4), -0.15)
    assert candidate_id("Q-learning beta=5", p, 43) == candidate_id("Q-learning beta=5", p.copy(), 43)
    assert candidate_id("Q-learning beta=5", p, 43) != candidate_id("Q-learning beta=5", p, 44)
    assert candidate_id("Monte Carlo", p) != candidate_id("Q-learning beta=5", p, 43)
    assert scenario_id("abc") == scenario_id("abc") and scenario_id("abc").startswith("S6-")


def test_exact_and_execution_duplicates() -> None:
    a = np.full((5, 4), -0.15)
    b = a.copy()
    b[4] = 0.0                                       # differs only on day 5
    assert sequence_key(a) != sequence_key(b)
    assert execution_key(a, 3) == execution_key(b, 3)  # breach on day 3: identical executed actions
    assert execution_key(a, 5) != execution_key(b, 5)
    assert sequence_key(a) == sequence_key(a + 1e-9)    # rounding removes float noise


@pytest.mark.parametrize("sums, expected", [
    ([-0.3, -0.3, -0.3, -0.3], "broad-market"),
    ([-0.3, -0.3, 0.0, 0.0], "broad-market"),
    ([-0.2, -0.5, 0.0, 0.0], "technology-led"),
    ([-0.1, -0.1, -0.6, -0.1], "financial-led"),
    ([-0.1, -0.1, -0.1, -0.6], "energy-led"),
    ([-0.1, -0.1, -0.5, -0.5], "sector-led (financials and energy jointly)"),
    ([-0.5, -0.5, 0.3, 0.0], "mixed-direction"),
    ([-0.05, -0.05, 0.0, 0.0], "unclassified"),
])
def test_family_rules(sums, expected) -> None:
    executed = np.array([sums])                     # one executed day
    assert classify_family(executed) == expected


def test_mixed_direction_within_a_factor_over_time() -> None:
    executed = np.array([[-0.3, -0.3, -0.3, -0.3], [0.0, 0.0, 0.0, 0.2]])
    assert classify_family(executed) == "mixed-direction"


def test_cumulative_feature_flat_after_breach_and_one_step_distance() -> None:
    a = np.full((5, 4), -0.15)
    f = cumulative_feature(a, 2).reshape(5, 4)
    assert np.allclose(f[1:], f[1]) and np.allclose(f[1], -0.30)
    b = a.copy()
    b[0, 0] -= 0.15
    d = np.linalg.norm(cumulative_feature(a, 5) - cumulative_feature(b, 5))
    assert d == pytest.approx(one_step_distance(0.15, 5))


def test_clustering_deterministic_and_threshold_monotone() -> None:
    rng = np.random.default_rng(0)
    centres = np.array([[0.0] * 20, [2.0] * 20, [-2.0] * 20])
    X = np.vstack([c + rng.normal(0, 0.05, (10, 20)) for c in centres])
    lab1, lab2 = cluster_features(X, 1.0), cluster_features(X.copy(), 1.0)
    assert np.array_equal(lab1, lab2) and len(set(lab1)) == 3
    sens = cluster_sensitivity(X, [0.01, 1.0, 100.0])
    assert list(sens["clusters"]) == sorted(sens["clusters"], reverse=True)
    assert sens["clusters"].iloc[-1] == 1 and len(cluster_features(np.zeros((0, 20)), 1.0)) == 0


def _rec(valid, breached, implausible, rank=0.05, dd=0.11, day=3):
    return SimpleNamespace(valid_breach=valid, breached=breached, implausible_breach=implausible,
                           historical_percentile=rank, path_drawdown=dd, breach_day=day)


def test_robustness_separates_valid_and_implausible() -> None:
    recs = [_rec(True, True, False)] * 8 + [_rec(False, True, True, rank=0.002)] + [_rec(False, False, False, day=None)]
    s = robustness_summary(recs)
    assert s["episodes"] == 10 and s["p_valid"] == 0.8 and s["p_implausible"] == 0.1 and s["p_any_breach"] == 0.9
    lo, hi = s["p_valid_ci95"]
    assert lo < 0.8 < hi and wilson_interval(0, 10)[0] == 0.0
    assert s["median_valid_rank"] == 0.05
    with pytest.raises(ValueError):
        robustness_summary([])


def test_holding_contributions_sum_exactly_and_factor_attribution() -> None:
    rng = np.random.default_rng(1)
    r = rng.normal(-0.02, 0.01, (3, 4))
    w = np.full(4, 0.25)
    out = holding_contributions(r, w, 100_000.0, ["A", "B", "C", "D"])
    values = 100_000.0 * np.concatenate([[1.0], np.cumprod(1 + r @ w)])
    assert out["pnl"].sum() == pytest.approx(values[-1] - values[0]) and out.attrs["final_value"] == pytest.approx(values[-1])
    assert out["share_of_loss"].sum() == pytest.approx(1.0)
    att = factor_attribution(np.full((3, 2), -0.01), np.ones((4, 2)), w, values, ["F1", "F2"])
    assert set(att) == {"F1", "F2"} and att["F1"] == pytest.approx(-0.01 * values[:3].sum())


def _scenario(status="validated", rank=0.05, p=0.9):
    s = {f: None for f in SCENARIO_FIELDS}
    s.update(scenario_id=scenario_id("x" + status + str(rank)), status=status, deterministic_rank=rank, screening_p_valid=p,
             executed_days=2, action_path=[[-0.15] * 4, [-0.15] * 4],
             robust_action_path=[[-0.15] * 4, [-0.15] * 4, [0.0] * 4, [0.0] * 4, [-0.15] * 4])
    return s


def test_library_schema_and_selection_rule() -> None:
    lib = {"schema_version": LIBRARY_SCHEMA_VERSION, "metadata": {}, "scenarios": [_scenario(), _scenario("exploratory", 0.02, 0.5)]}
    validate_library(lib, q=0.01, p_valid_target=0.8)
    assert json.loads(to_json({"x": np.float64("nan"), "y": np.int64(3), "z": np.array([1.0])})) == {"x": None, "y": 3, "z": [1.0]}
    bad = {**lib, "scenarios": [_scenario(p=0.5)]}                  # "validated" but below the rule
    with pytest.raises(ValueError):
        validate_library(bad, q=0.01, p_valid_target=0.8)
    dup = {**lib, "scenarios": [_scenario(), _scenario()]}
    with pytest.raises(ValueError):
        validate_library(dup, q=0.01, p_valid_target=0.8)
    shape = _scenario()
    shape["action_path"] = [[-0.15] * 3]
    with pytest.raises(ValueError):
        validate_library({**lib, "scenarios": [shape]}, q=0.01, p_valid_target=0.8)


def test_library_requires_full_robustness_path_consistent_with_executed_days() -> None:
    lib = {"schema_version": LIBRARY_SCHEMA_VERSION, "metadata": {}, "scenarios": [_scenario()]}
    validate_library(lib, q=0.01, p_valid_target=0.8)
    mism = _scenario()
    mism["robust_action_path"] = [[-0.30] * 4] * 5                       # does not start with the executed path
    with pytest.raises(ValueError):
        validate_library({**lib, "scenarios": [mism]}, q=0.01, p_valid_target=0.8)
    short = _scenario()
    short["robust_action_path"] = [[-0.15] * 4]                            # shorter than the executed days
    with pytest.raises(ValueError):
        validate_library({**lib, "scenarios": [short]}, q=0.01, p_valid_target=0.8)
