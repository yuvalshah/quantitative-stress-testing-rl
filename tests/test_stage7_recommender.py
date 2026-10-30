"""Stage 7 recommender: loading, eligibility, scoring, ranking, diversity, explanations, profile."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rlstress.scenario_library import LIBRARY_SCHEMA_VERSION, SCENARIO_FIELDS, to_json
from rlstress.scenario_recommender import (
    COMPONENTS,
    RecommenderConfig,
    coverage_gaps,
    eligibility,
    explain,
    load_library,
    portfolio_profile,
    relevance,
    scenario_features,
    score,
    select_diverse,
    select_random,
    select_top,
    shortlist_metrics,
)

SD = np.array([0.0106, 0.0133, 0.0141, 0.0195])
BOUNDS = SD * 12


def _scen(sid, actions, status="validated", rank=0.05, p=0.95, imp=0.0, n=500, family="broad-market", cluster=1, scr=None):
    a = np.asarray(actions, dtype=float)
    s = {f: None for f in SCENARIO_FIELDS}
    s.update(scenario_id=sid, status=status, selection_reason="test", family=family, cluster=cluster, discovery_methods=["Monte Carlo (discrete)"],
             provenance={"cluster_members": 3}, representative_candidate_id=f"cand-{sid}", executed_days=len(a), action_path=a.tolist(),
             factor_shock_path=(a * BOUNDS).tolist(), robust_action_path=np.vstack([a, np.zeros((5 - len(a), 4))]).tolist(),
             robust_factor_shock_path=(np.vstack([a, np.zeros((5 - len(a), 4))]) * BOUNDS).tolist(), deterministic_rank=rank, deterministic_drawdown=0.11, deterministic_breach_day=len(a),
             p_valid=p, p_valid_ci95=None if p is None else [max(0.0, p - 0.03), min(1.0, p + 0.03)], p_any_breach=p, p_implausible=imp,
             mean_drawdown_at_breach=0.11, mean_breach_day=4.0, screening_p_valid=p if scr is None else scr, screening_episodes=200,
             reporting_episodes=n, vulnerability={"holding_pnl": {"AAPL": -2000.0}})
    return s


def _library(scenarios):
    return {"schema_version": LIBRARY_SCHEMA_VERSION, "scenarios": scenarios,
            "metadata": {"plausibility_threshold_q": 0.01, "selection_rule": {"p_valid_target": 0.8},
                         "factor_order": ["SPY_return", "QQQ_return", "XLF_return", "XLE_return"]}}


def _profile():
    B = np.tile([1.0, 0.2, 0.3, 0.1], (8, 1))
    rets = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (300, 8)), columns=list("ABCDEFGH"))
    return portfolio_profile("test", list("ABCDEFGH"), [0.125] * 8, 100_000.0, B, SD, {t: "Tech" for t in "ABCD"}, rets, 0.10)


def _scored(lib, cfg=None, impacts=None):
    cfg = cfg or RecommenderConfig()
    f = eligibility(scenario_features(lib, SD, 5), cfg)
    impacts = impacts if impacts is not None else {s["scenario_id"]: 0.10 + 0.01 * i for i, s in enumerate(lib["scenarios"])}
    return score(f, _profile(), impacts, cfg), cfg


def test_config_validation() -> None:
    RecommenderConfig()
    with pytest.raises(ValueError):
        RecommenderConfig(weights={"relevance": 0.5, "impact": 0.5, "plausibility": 0.5, "robustness": -0.5})
    with pytest.raises(ValueError):
        RecommenderConfig(weights={"relevance": 1.0})
    with pytest.raises(ValueError):
        RecommenderConfig(top_k=0)


def test_load_library_missing_malformed_empty(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_library(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "other", "metadata": {}, "scenarios": []}))
    with pytest.raises(ValueError):
        load_library(bad)
    empty = tmp_path / "empty.json"
    empty.write_text(to_json(_library([])))
    with pytest.raises(ValueError):
        load_library(empty)
    ok = tmp_path / "ok.json"
    ok.write_text(to_json(_library([_scen("S6-a", [[-0.3] * 4, [-0.15] * 4])])))
    assert load_library(ok)["scenarios"][0]["scenario_id"] == "S6-a"


def test_missing_evidence_and_filters_make_scenarios_ineligible() -> None:
    lib = _library([_scen("ok", [[-0.3] * 4]), _scen("expl", [[-0.3] * 4], status="exploratory", scr=0.6),
                    _scen("nop", [[-0.3] * 4], p=None), _scen("imp", [[-0.3] * 4], imp=0.2), _scen("few", [[-0.3] * 4], n=20)])
    f = eligibility(scenario_features(lib, SD, 5), RecommenderConfig()).set_index("scenario_id")
    assert f.loc["ok", "eligible"] and not f.loc[["expl", "nop", "imp", "few"], "eligible"].any()
    assert "not validated" in f.loc["expl", "ineligible_reason"] and "missing p_valid" in f.loc["nop", "ineligible_reason"]
    assert "P(implausible)" in f.loc["imp", "ineligible_reason"] and "reporting episodes" in f.loc["few", "ineligible_reason"]
    assert np.isnan(f.loc["nop", "p_valid"])                                        # missing stays missing


def test_scores_normalised_weighted_and_missing_impact() -> None:
    lib = _library([_scen(f"s{i}", [[-0.1 - 0.05 * i] * 4]) for i in range(4)])
    scored, cfg = _scored(lib, impacts={"s0": 0.10, "s1": 0.12, "s2": 0.14})       # s3 lacks impact
    e = scored[scored["eligible"]]
    assert set(e["scenario_id"]) == {"s0", "s1", "s2"} and scored.set_index("scenario_id").loc["s3", "ineligible_reason"] == "missing portfolio impact"
    for c in COMPONENTS:
        assert e[c].between(0, 1).all()
    assert e["impact"].min() == 0 and e["impact"].max() == 1
    assert np.allclose(e["score"], sum(0.25 * e[c] for c in COMPONENTS))
    again, _ = _scored(lib, impacts={"s0": 0.10, "s1": 0.12, "s2": 0.14})
    assert again["score"].equals(scored["score"])                                 # deterministic


def test_relevance_direction() -> None:
    p = _profile()
    aligned = -p.loss_direction * -1                                                # same direction as the loss direction
    assert relevance(aligned, p) == pytest.approx(1.0) and relevance(-aligned, p) == pytest.approx(0.0)
    assert relevance(np.zeros(4), p) == 0.5


def test_ranking_diversity_and_random_baseline() -> None:
    near = [[-0.30] * 4, [-0.15] * 4]
    lib = _library([_scen("a", near, rank=0.08, cluster=1), _scen("b", [[-0.30] * 4, [-0.16] * 4], rank=0.079, cluster=1),
                    _scen("c", [[-0.1, -0.1, -0.4, -0.4], [-0.1, -0.1, -0.2, -0.2]], rank=0.03, cluster=2, family="sector-led (financials and energy jointly)")])
    scored, cfg = _scored(lib, impacts={"a": 0.12, "b": 0.12, "c": 0.12})
    top = select_top(scored, 2)
    assert top == select_top(scored, 2) and top[0] == "a"
    no_pen = select_diverse(scored, 2, RecommenderConfig(diversity_penalty=0.0))
    assert [c["scenario_id"] for c in no_pen] == top                              # lambda = 0 reproduces pure ranking
    div = select_diverse(scored, 2, RecommenderConfig(diversity_penalty=5.0))
    assert [c["scenario_id"] for c in div] == ["a", "c"]                          # near-duplicate b is skipped
    m_top, m_div = shortlist_metrics(scored, top, cfg), shortlist_metrics(scored, [c["scenario_id"] for c in div], cfg)
    assert m_div["distinct clusters"] > m_top["distinct clusters"] and m_div["min pairwise distance"] > m_top["min pairwise distance"]
    r1, r2 = select_random(scored, 2, seed=1), select_random(scored, 2, seed=1)
    assert r1 == r2 and set(r1) <= set(scored.loc[scored["eligible"], "scenario_id"])
    assert len(select_top(scored, 10)) == 3 and len(select_diverse(scored, 10, cfg)) == 3   # fewer eligible than k


def test_empty_eligible_set_and_coverage_gaps() -> None:
    lib = _library([_scen("x", [[-0.3] * 4], status="exploratory", scr=0.5)])
    scored, cfg = _scored(lib)
    assert select_top(scored, 3) == [] and select_diverse(scored, 3, cfg) == [] and select_random(scored, 3, 0) == []
    assert shortlist_metrics(scored, [], cfg)["size"] == 0
    gaps = coverage_gaps(scored).set_index("family")
    assert gaps.loc["broad-market", "in library"] and not gaps.loc["broad-market", "eligible"] and not gaps.loc["technology-led", "in library"]


def test_provenance_and_explanation_grounded() -> None:
    lib = _library([_scen("a", [[-0.3] * 4, [-0.15] * 4]), _scen("b", [[-0.1, -0.1, -0.4, -0.4]], family="sector-led (financials and energy jointly)", cluster=2)])
    scored, cfg = _scored(lib, impacts={"a": 0.12, "b": 0.11})
    rows = scored.set_index("scenario_id", drop=False)
    assert rows.loc["a", "representative_candidate_id"] == "cand-a" and rows.loc["a", "discovery_methods"] == "Monte Carlo (discrete)"
    text = explain(rows.loc["b"], _profile(), [rows.loc["a"]], cfg, {"AAPL": -2000.0})
    assert "rank 0.0500" in text and "500 reporting episodes" in text and "nearest a" in text and "AAPL" in text


def test_portfolio_profile_validation() -> None:
    p = _profile()
    assert set(p.factor_loadings) == {"SPY", "QQQ", "XLF", "XLE"} and np.isclose(np.linalg.norm(p.loss_direction), 1.0)
    assert p.sector_weights["Tech"] == pytest.approx(0.5) and p.max_drawdown <= 0
    with pytest.raises(ValueError):
        portfolio_profile("bad", list("ABCDEFGH"), [0.5] * 8, 1.0, np.ones((8, 4)), SD, {}, pd.DataFrame(np.zeros((5, 8)), columns=list("ABCDEFGH")), 0.1)


def test_exact_weight_shift() -> None:
    from rlstress.scenario_recommender import exact_weight_shift
    base = {c: 0.25 for c in COMPONENTS}
    up = exact_weight_shift(base, "impact", 0.05)
    assert up["impact"] == pytest.approx(0.30) and all(up[c] == pytest.approx(0.25 - 0.05 / 3) for c in COMPONENTS if c != "impact")
    down = exact_weight_shift(base, "impact", -0.05)
    assert down["impact"] == pytest.approx(0.20) and sum(down.values()) == pytest.approx(1.0)
    RecommenderConfig(weights=up)                                                   # valid config
    with pytest.raises(ValueError):
        exact_weight_shift(base, "impact", -0.30)
    with pytest.raises(ValueError):
        exact_weight_shift(base, "nope", 0.05)


def test_shortlist_fresh_metrics() -> None:
    from rlstress.scenario_recommender import shortlist_fresh_metrics
    fresh = pd.DataFrame({"p_valid": [0.9, 0.8, 1.0], "p_implausible": [0.0, 0.1, 0.0], "mean_drawdown_at_breach": [0.11, 0.12, 0.13],
                          "episodes": [500] * 3}, index=["a", "b", "c"])
    m = shortlist_fresh_metrics(fresh, ["a", "b"])
    assert m["fresh mean P(valid)"] == pytest.approx(0.85) and m["fresh min P(valid)"] == 0.8 and m["fresh mean P(implausible)"] == pytest.approx(0.05)
    assert shortlist_fresh_metrics(fresh, [])["size"] == 0
