"""Stage 5 policies, evaluation metrics and threshold calibration (no stable-baselines3 needed)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rlstress.env import make_env
from rlstress.rl.calibration import real_window_percentiles, select_threshold, threshold_calibration
from rlstress.rl.evaluation import (
    EpisodeRecord,
    across_sets,
    diversity,
    evaluate_policy,
    hacking_diagnostics,
    run_episode,
    summarize,
)
from rlstress.rl.policies import FrozenObsNormalizer, RandomPolicy, scripted_decline_policy, zero_policy
from rlstress.shocks import fit_shock_model, historical_path_percentile


def _rec(**kw) -> EpisodeRecord:
    base = dict(policy="p", episode_seed=0, anchor_date="2017-01-03", length=5, total_reward=0.0,
                components={"breach": 0.0, "severity": 0.0, "plausibility": 0.0, "magnitude": 0.0,
                            "step": -0.05, "invalid": 0.0},
                termination="horizon", breached=False, valid_breach=False, implausible_breach=False,
                drawdown_breach=False, loss_breach=False, cumulative_loss=0.0, path_drawdown=0.0,
                breach_day=None, historical_percentile=1.0, path_log_tail_probability=0.0,
                mean_abs_action=0.0, mean_sq_action=0.0, saturated_action_share=0.0, residual_draws=1.0,
                actions=[[0.0] * 4] * 5, factor_shocks=[[0.0] * 4] * 5)
    base.update(kw)
    return EpisodeRecord(**base)


def _valid(dd: float, day: int, shocks: list[list[float]], actions=None) -> EpisodeRecord:
    return _rec(termination="valid_breach", breached=True, valid_breach=True, drawdown_breach=True,
                path_drawdown=dd, breach_day=day, total_reward=9.0,
                components={"breach": 10.0, "severity": 0.5, "plausibility": -1.4, "magnitude": -0.05,
                            "step": -0.05, "invalid": 0.0},
                factor_shocks=shocks, actions=actions or [[-0.2] * 4] * day)


# --- Metrics on hand-constructed records ------------------------------------------------------

def test_summarize_hand_example() -> None:
    recs = [
        _valid(0.12, 5, [[-0.02] * 4] * 5),
        _valid(0.10, 4, [[-0.03] * 4] * 4),
        _rec(termination="implausible_breach", breached=True, implausible_breach=True, drawdown_breach=True,
             path_drawdown=0.2, breach_day=1, total_reward=-2.0),
        _rec(total_reward=-0.05),
    ]
    s = summarize(recs)
    assert s["n_episodes"] == 4
    assert s["breach_rate"] == 0.75 and s["valid_breach_rate"] == 0.5 and s["implausible_breach_rate"] == 0.25
    assert s["valid_share_of_breaches"] == pytest.approx(2 / 3)
    assert s["severity_valid_breaches"]["mean"] == pytest.approx(0.11)
    assert s["severity_breaches"]["median"] == pytest.approx(0.12)
    assert s["breach_day_distribution"] == {"1": 1, "2": 0, "3": 0, "4": 1, "5": 1}
    assert s["reward_components_mean"]["breach"] == pytest.approx(5.0)
    assert s["episode_return"]["mean"] == pytest.approx((9 + 9 - 2 - 0.05) / 4)
    assert s["loss_breach_without_drawdown_breach"] == 0


def test_diversity_detects_collapse() -> None:
    same = [_valid(0.11, 5, [[-0.02] * 4] * 5) for _ in range(4)]
    d = diversity(same, np.full(4, 0.01), ["SPY", "QQQ", "XLF", "XLE"])
    assert d["unique_action_sequences_share"] == 0.25
    assert d["mean_pairwise_distance_sd_units"] == pytest.approx(0.0)
    assert d["all_factors_negative_share"] == 1.0
    varied = same[:2] + [_valid(0.11, 5, [[-0.04, -0.01, 0.0, -0.03]] * 5, [[-0.5, 0.1, 0.2, -0.3]] * 5)]
    assert diversity(varied, np.full(4, 0.01), ["SPY", "QQQ", "XLF", "XLE"])["mean_pairwise_distance_sd_units"] > 0
    assert diversity(same[:1], np.full(4, 0.01), ["a", "b", "c", "d"])["n_valid"] == 1


def test_hacking_diagnostics_flags_unearned_breach_reward() -> None:
    bad = _rec(components={"breach": 10.0, "severity": 0, "plausibility": 0, "magnitude": 0, "step": 0, "invalid": 0},
               termination="implausible_breach", breached=True, implausible_breach=True, breach_day=1)
    h = hacking_diagnostics([bad, _valid(0.1, 5, [[-0.02] * 4] * 5)])
    assert h["breach_reward_paid_without_valid_breach"] == 1
    assert h["day1_breach_share_of_breaches"] == 0.5


def test_across_sets() -> None:
    s1 = summarize([_valid(0.1, 5, [[-0.02] * 4] * 5), _rec()])
    s2 = summarize([_rec(), _rec()])
    a = across_sets([s1, s2])
    assert a["valid_breach_rate"]["mean"] == pytest.approx(0.25)
    assert a["valid_breach_rate"]["std"] == pytest.approx(np.std([0.5, 0.0], ddof=1))


# --- Policies --------------------------------------------------------------------------------

def test_baseline_policies_produce_valid_actions() -> None:
    assert np.all(zero_policy(4).act(None) == 0)
    assert np.all(scripted_decline_policy(4, -0.15).act(None) == np.float32(-0.15))
    r = RandomPolicy(4, offset=7)
    r.reset(3)
    a1 = [r.act(None) for _ in range(5)]
    r.reset(3)
    a2 = [r.act(None) for _ in range(5)]
    np.testing.assert_array_equal(np.array(a1), np.array(a2))
    assert np.all(np.abs(np.array(a1)) <= 1)


def test_frozen_normalizer_matches_vecnormalize_formula() -> None:
    n = FrozenObsNormalizer(np.array([1.0, -2.0]), np.array([4.0, 0.25]), clip_obs=10.0, epsilon=1e-8)
    out = n.normalize_obs(np.array([3.0, 100.0]))
    assert out[0] == pytest.approx(1.0, rel=1e-6)
    assert out[1] == 10.0  # clipped


# --- Evaluation on the real environment (synthetic data) ---------------------------------------

def test_common_random_numbers_and_record_consistency(s4_source, s4_config) -> None:
    env = make_env(s4_config, s4_source)
    seeds = range(100, 110)
    z = evaluate_policy(env, zero_policy(4), seeds)
    sc = evaluate_policy(env, scripted_decline_policy(4, -0.15), seeds)
    assert [r.anchor_date for r in z] == [r.anchor_date for r in sc]
    for r in z + sc:
        assert r.total_reward == pytest.approx(sum(r.components.values()))
        assert r.length == len(r.actions) == len(r.factor_shocks)
        if r.breached:
            assert r.breach_day == r.length and r.drawdown_breach
    assert not any(r.breached for r in z)


def test_implausible_breach_never_counted_valid(s4_source, s4_config) -> None:
    env = make_env(s4_config, s4_source)
    r = run_episode(env, scripted_decline_policy(4, -1.0), 5)
    assert r.breached and r.implausible_breach and not r.valid_breach
    assert r.components["breach"] == 0.0 and r.termination == "implausible_breach"


def test_threshold_semantics_match_stage4(s4_source, s4_config) -> None:
    strict = dataclasses.replace(s4_config, simulation=dataclasses.replace(s4_config.simulation, min_plausibility=0.5))
    loose = dataclasses.replace(s4_config, simulation=dataclasses.replace(s4_config.simulation, min_plausibility=0.001))
    pol = scripted_decline_policy(4, -0.4)
    r_strict = run_episode(make_env(strict, s4_source), pol, 1)
    r_loose = run_episode(make_env(loose, s4_source), pol, 1)
    assert r_strict.breached and r_loose.breached
    assert r_strict.historical_percentile == r_loose.historical_percentile
    assert r_strict.valid_breach == (r_strict.historical_percentile >= 0.5)
    assert r_loose.valid_breach == (r_loose.historical_percentile >= 0.001)


def test_real_window_calibration_on_training_is_nominal(s4_source, s4_config) -> None:
    s, e = s4_config.data.splits.train
    m = fit_shock_model(s4_source, s4_config.portfolio.tickers, s4_config.simulation, s, e)
    pct = real_window_percentiles(m, s4_source, s, e, 5)
    assert len(pct) == m.n_obs - 4
    for q in (0.01, 0.05, 0.2):
        assert (pct < q).mean() <= q + 1e-9  # by construction on the reference windows
    calib = threshold_calibration(m, s4_source, {"train": (s, e), "validation": s4_config.data.splits.validation},
                                  [0.01, 0.05], 5)
    assert set(calib) >= {"train", "validation", "validation_calibration_ratio", "semantics"}


def test_select_threshold_rule() -> None:
    calib = {"validation": {"rejected_share": {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0}}}
    assert select_threshold(calib, [0.005, 0.01, 0.02], 0.01)["selected"] == 0.01  # all tie -> default
    calib = {"validation": {"rejected_share": {"0.005": 0.03, "0.01": 0.011, "0.02": 0.05}}}
    assert select_threshold(calib, [0.005, 0.01, 0.02], 0.02)["selected"] == 0.01
