"""Stage 4 Gymnasium environment: episodes, termination, reward, determinism, demo smoke test."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rlstress.config import PROJECT_ROOT, config_from_dict
from rlstress.env import make_env
from rlstress.shocks import InvalidActionError

from conftest import stage4_raw_config

ANCHOR = "2017-06-01"


def _mean_mode(config):
    return dataclasses.replace(config, simulation=dataclasses.replace(config.simulation, residual_mode="mean"))


def _run(env, action, seed=0, anchor=ANCHOR):
    env.reset(seed=seed, options={"anchor_date": anchor})
    history = []
    for _ in range(env.horizon):
        obs, r, term, trunc, info = env.step(np.asarray(action, dtype=np.float32))
        history.append((obs, r, term, trunc, info))
        if term:
            break
    return history


def test_gymnasium_api_compliance(s4_source, s4_config) -> None:
    env = make_env(s4_config, s4_source)
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
    assert len(env.observation_names) == obs.shape[0]
    assert env.action_space.shape == (4,)


def test_zero_shock_episode(s4_source, s4_config) -> None:
    env = make_env(_mean_mode(s4_config), s4_source)
    hist = _run(env, np.zeros(4))
    assert len(hist) == env.horizon
    obs, r, term, trunc, info = hist[-1]
    assert term and not trunc and not info["breached"]
    assert abs(info["cumulative_loss"]) < 0.01 and info["path_drawdown"] < 0.01
    assert info["path_historical_percentile"] > 0.5
    assert info["reward_components"]["breach"] == 0.0


def test_broad_crash_terminates_on_drawdown_breach(s4_source, s4_config) -> None:
    env = make_env(_mean_mode(s4_config), s4_source)
    hist = _run(env, np.full(4, -0.6))
    obs, r, term, trunc, info = hist[-1]
    assert term and info["breached"] and len(hist) < env.horizon  # early termination
    dd = next(x for x in info["limit_results"] if x["metric"] == "max_drawdown")
    assert dd["breached"] and dd["observed"] >= 0.10
    loss = next(x for x in info["limit_results"] if x["metric"] == "cumulative_loss")
    assert loss["status"] == "not_evaluated" and loss["breached"] is None  # loss only judged at the horizon


def test_loss_limit_checked_only_at_horizon(s4_source) -> None:
    raw = stage4_raw_config()
    raw["risk"]["limits"] = [l for l in raw["risk"]["limits"] if l["name"] == "stress_loss_5d"]
    env = make_env(_mean_mode(config_from_dict(raw)), s4_source)
    hist = _run(env, np.full(4, -0.6))
    assert len(hist) == env.horizon  # no early stop without a drawdown limit
    info = hist[-1][4]
    assert info["breached"] and info["limit_results"][0]["metric"] == "cumulative_loss"
    assert info["cumulative_loss"] >= 0.10


def test_plausible_vs_implausible_breach_reward(s4_source, s4_config) -> None:
    env = make_env(_mean_mode(s4_config), s4_source)
    moderate = None
    for c in np.arange(0.05, 1.0, 0.025):  # smallest coherent crash that breaches
        hist = _run(env, np.full(4, -c))
        if hist[-1][4]["breached"]:
            moderate = hist
            break
    assert moderate is not None and moderate[-1][4]["valid_breach"]
    assert moderate[-1][4]["reward_components"]["breach"] == s4_config.environment.breach_reward
    extreme = _run(env, np.full(4, -1.0))[-1][4]
    assert extreme["breached"] and extreme["implausible_breach"] and not extreme["valid_breach"]
    assert extreme["reward_components"]["breach"] == 0.0


def test_reward_components_sum_and_signs(s4_source, s4_config) -> None:
    env = make_env(s4_config, s4_source)
    env.reset(seed=3)
    _, r, _, _, info = env.step(np.array([-0.3, -0.3, 0.2, -0.1], dtype=np.float32))
    comps = info["reward_components"]
    assert r == pytest.approx(sum(comps.values()))
    assert comps["plausibility"] <= 0 and comps["magnitude"] <= 0 and comps["step"] < 0
    assert comps["magnitude"] == pytest.approx(-s4_config.environment.magnitude_penalty * np.mean(np.array([-0.3, -0.3, 0.2, -0.1]) ** 2), rel=1e-5)
    assert set(comps) == {"breach", "severity", "plausibility", "magnitude", "step", "invalid", "plausibility_credit"}
    assert comps["plausibility_credit"] == 0.0                         # default kappa = 0: original reward


def test_deterministic_with_fixed_seed(s4_source, s4_config) -> None:
    env = make_env(s4_config, s4_source)
    def trajectory(seed):
        obs, _ = env.reset(seed=seed)
        out = [obs]
        for a in ([-0.2, -0.1, 0, 0.1], [0.1, -0.3, -0.2, 0], [0, 0, -0.5, 0.2]):
            obs, r, term, _, _ = env.step(np.array(a, dtype=np.float32))
            out += [obs, np.array([r])]
            if term:
                break
        return np.concatenate(out)
    np.testing.assert_array_equal(trajectory(11), trajectory(11))
    assert not np.array_equal(trajectory(11), trajectory(12))


def test_invalid_actions_and_lifecycle(s4_source, s4_config) -> None:
    env = make_env(s4_config, s4_source)
    env.reset(seed=0)
    for bad in (np.zeros(3), np.array([np.nan, 0, 0, 0]), np.array([1.5, 0, 0, 0])):
        with pytest.raises(InvalidActionError):
            env.step(bad)
    env.step(np.ones(4, dtype=np.float32))  # boundary action is valid
    fresh = make_env(s4_config, s4_source)
    with pytest.raises(RuntimeError, match="reset"):
        fresh.step(np.zeros(4))


def test_invalid_simulation_terminates_with_penalty(s4_source, s4_config) -> None:
    cfg = dataclasses.replace(s4_config, simulation=dataclasses.replace(s4_config.simulation, max_abs_stock_return=0.01))
    env = make_env(cfg, s4_source)
    env.reset(seed=0)
    _, r, term, _, info = env.step(np.full(4, -1.0, dtype=np.float32))
    assert term and info["invalid"]
    assert info["reward_components"]["invalid"] == -cfg.environment.invalid_penalty


def test_anchor_states_from_anchor_window_only(s4_source, s4_config) -> None:
    env = make_env(s4_config, s4_source)
    assert env.market_states.index.max() <= np.datetime64("2018-12-31")
    with pytest.raises(ValueError):
        env.reset(options={"anchor_date": "2020-03-02"})  # test-split date


def test_stage3_result_available_for_episode(s4_source, s4_config) -> None:
    env = make_env(_mean_mode(s4_config), s4_source)
    _run(env, np.full(4, -0.6))
    res = env.current_result()
    assert res.scenario_type == "simulated_factor_shock" and res.breached
    assert env.scenario().n_steps == res.horizon_days


def test_horizon_without_matching_limit_rejected(s4_source) -> None:
    raw = stage4_raw_config()
    raw["simulation"]["horizon_days"] = 3
    with pytest.raises(ValueError, match="horizon"):
        make_env(config_from_dict(raw), s4_source)


def test_stage4_demo_smoke(tmp_path, s4_source, s4_config) -> None:
    path = tmp_path / "consolidated_daily.csv"
    from rlstress.data.storage import write_frame
    write_frame(s4_source, path)
    spec = importlib.util.spec_from_file_location("s4demo", PROJECT_ROOT / "examples" / "stage4_shock_model_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.run_demo(s4_config, path, tmp_path / "out", baseline_episodes=5)
    for f in out["outputs"].values():
        assert Path(f).exists()
    assert out["diagnostics"]["fit_window"]["actual"][1] <= "2018-12-31"
    assert {e["episode"] for e in out["episodes"]["scripted_episodes"]} >= {"zero_shock"}
    with pytest.raises(FileExistsError):
        mod.run_demo(s4_config, path, tmp_path / "out", baseline_episodes=5)


REAL = PROJECT_ROOT / "data" / "processed" / "features" / "consolidated_daily.csv"


@pytest.mark.skipif(not REAL.exists(), reason="Stage 2 real data not available locally")
def test_real_data_integration() -> None:
    from rlstress.config import load_config
    from rlstress.data.storage import read_frame

    config = load_config()
    env = make_env(config, read_frame(REAL))
    m = env.simulator.model
    assert m.fit_end <= config.data.splits.train[1] and m.n_obs > 1500
    assert m.nu in config.simulation.dof_grid
    check_env(env, skip_render_check=True)


# --- Implausible-breach penalty (fix for the Stage 5 "implausible-crash trap") ---------------

def _with_penalty(config, penalty):
    return dataclasses.replace(_mean_mode(config), environment=dataclasses.replace(config.environment,
                                                                                    implausible_breach_penalty=penalty))


def test_implausible_breach_penalty_default_is_legacy(s4_source, s4_config) -> None:
    assert s4_config.environment.implausible_breach_penalty == 0.0
    legacy = _run(make_env(_mean_mode(s4_config), s4_source), np.full(4, -1.0))
    explicit = _run(make_env(_with_penalty(s4_config, 0.0), s4_source), np.full(4, -1.0))
    assert [h[1] for h in legacy] == [h[1] for h in explicit]          # identical rewards step by step


def test_implausible_breach_penalty_charged_once_and_only_on_implausible(s4_source, s4_config) -> None:
    penalty = 5.0
    env = make_env(_with_penalty(s4_config, penalty), s4_source)
    extreme = _run(env, np.full(4, -1.0))
    last = extreme[-1][4]
    assert last["implausible_breach"] and not last["valid_breach"]
    assert last["reward_components"]["breach"] == -penalty
    assert sum(h[4]["reward_components"]["breach"] for h in extreme) == -penalty
    assert extreme[-1][1] == pytest.approx(sum(last["reward_components"].values()))
    for c in np.arange(0.05, 1.0, 0.025):                               # a valid breach is unaffected
        hist = _run(env, np.full(4, -c))
        if hist[-1][4]["breached"]:
            assert hist[-1][4]["valid_breach"]
            assert hist[-1][4]["reward_components"]["breach"] == s4_config.environment.breach_reward
            break
    calm = _run(env, np.zeros(4))                                         # no breach: no charge
    assert all(h[4]["reward_components"]["breach"] == 0.0 for h in calm)


def test_implausible_breach_penalty_charged_once_without_termination(s4_source, s4_config) -> None:
    cfg = _with_penalty(s4_config, 5.0)
    cfg = dataclasses.replace(cfg, environment=dataclasses.replace(cfg.environment, terminate_on_breach=False))
    hist = _run(make_env(cfg, s4_source), np.full(4, -1.0))
    assert len(hist) == cfg.simulation.horizon_days
    assert sum(h[4]["reward_components"]["breach"] for h in hist) == -5.0


def test_negative_implausible_breach_penalty_rejected() -> None:
    from conftest import stage4_raw_config
    from rlstress.config import ConfigError

    raw = stage4_raw_config()
    raw["environment"]["implausible_breach_penalty"] = -1.0
    with pytest.raises(ConfigError):
        config_from_dict(raw)


# --- Graded plausibility credit (implausible_breach_credit = kappa) ---------------------------

def _with_credit(config, kappa, penalty=0.0):
    env_cfg = dataclasses.replace(config.environment, implausible_breach_credit=kappa, implausible_breach_penalty=penalty)
    return dataclasses.replace(_mean_mode(config), environment=env_cfg)


def _crash_episode(env, c, follow=-0.05):
    env.reset(seed=0, options={"anchor_date": ANCHOR})
    total, comps = 0.0, {}
    for d in range(env.horizon):
        _, r, term, _, info = env.step(np.full(4, -c if d == 0 else follow, dtype=np.float32))
        total += r
        for k, v in info["reward_components"].items():
            comps[k] = comps.get(k, 0.0) + v
        if term:
            break
    return total, comps, info


def test_log_threshold_reproduces_the_percentile_gate(s4_source, s4_config) -> None:
    from rlstress.shocks import historical_path_percentile, path_log_tail_probability, plausibility_log_threshold

    env = make_env(s4_config, s4_source)
    m = env.simulator.model
    rng = np.random.default_rng(1)
    for q in (0.005, 0.01, 0.02, 0.1):
        for _ in range(500):
            t = int(rng.integers(1, env.horizon + 1))
            s = list(rng.exponential(rng.uniform(0.2, 5.0), t))
            gate = historical_path_percentile(m, s) >= q
            assert gate == (path_log_tail_probability(s) >= plausibility_log_threshold(m, t, q))
    assert set(env.log_thresholds) == set(range(1, env.horizon + 1)) and all(v < 0 for v in env.log_thresholds.values())


def test_credit_fraction_boundaries() -> None:
    from rlstress.shocks import implausibility_credit_fraction as g

    assert g(-3.0, -4.0) == 1.0 and g(-4.0, -4.0) == 1.0               # at or above the threshold
    assert g(-8.0, -4.0) == pytest.approx(0.5) and 0.0 < g(-400.0, -4.0) < 0.011
    assert g(-np.inf, -4.0) == 0.0 and g(np.nan, -4.0) == 0.0           # underflow / undefined -> no credit
    xs = np.linspace(-4.0, -60.0, 200)
    vals = [g(x, -4.0) for x in xs]
    assert all(a >= b for a, b in zip(vals, vals[1:])) and all(0.0 <= v <= 1.0 for v in vals)


def test_graded_credit_default_is_legacy(s4_source, s4_config) -> None:
    for c in (1.0, 0.3, 0.0):
        legacy = _crash_episode(make_env(_mean_mode(s4_config), s4_source), c)
        zero = _crash_episode(make_env(_with_credit(s4_config, 0.0), s4_source), c)
        assert legacy[0] == zero[0] and zero[1]["plausibility_credit"] == 0.0


def test_graded_credit_monotone_bounded_and_dominated_by_valid(s4_source, s4_config) -> None:
    kappa = 0.5
    env = make_env(_with_credit(s4_config, kappa), s4_source)
    B = s4_config.environment.breach_reward
    implausible, valid = [], []
    for c in np.arange(1.0, 0.0, -0.025):
        total, comps, info = _crash_episode(env, float(c))
        assert total == pytest.approx(sum(comps.values()))                 # components reconcile with the return
        if info["implausible_breach"]:
            assert 0.0 < comps["plausibility_credit"] <= kappa * B and comps["breach"] == 0.0
            implausible.append((c, comps["plausibility_credit"], total))
        else:
            assert comps["plausibility_credit"] == 0.0                       # valid breach or no breach: no credit
            if info["valid_breach"]:
                assert comps["breach"] == B
                valid.append(total)
    assert len(implausible) >= 3 and valid
    credits = [x[1] for x in implausible]                                  # milder crash (smaller c) -> more credit
    assert all(a <= b + 1e-12 for a, b in zip(credits, credits[1:]))
    assert min(valid) > max(x[2] for x in implausible)                     # every valid episode beats every implausible one
    calm_total, calm_comps, calm_info = _crash_episode(env, 0.0, follow=0.0)
    assert not calm_info["breached"] and calm_comps["plausibility_credit"] == 0.0


def test_graded_credit_paid_once_and_combines_with_penalty(s4_source, s4_config) -> None:
    cfg = _with_credit(s4_config, 0.5, penalty=2.0)
    cfg = dataclasses.replace(cfg, environment=dataclasses.replace(cfg.environment, terminate_on_breach=False))
    hist = _run(make_env(cfg, s4_source), np.full(4, -1.0))
    credits = [h[4]["reward_components"]["plausibility_credit"] for h in hist]
    assert sum(c > 0 for c in credits) == 1                                # charged on the first implausible breach only
    assert sum(h[4]["reward_components"]["breach"] for h in hist) == -2.0


@pytest.mark.parametrize("kappa", [-0.1, 1.0, 1.5])
def test_invalid_credit_rejected(kappa) -> None:
    from conftest import stage4_raw_config
    from rlstress.config import ConfigError

    raw = stage4_raw_config()
    raw["environment"]["implausible_breach_credit"] = kappa
    with pytest.raises(ConfigError):
        config_from_dict(raw)
