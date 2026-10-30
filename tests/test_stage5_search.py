"""Stage 5 search utilities: grouped actions, tabular Q-learning, restarted CEM, path evaluation."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rlstress.env import make_env
from rlstress.search import (
    CEMConfig,
    GroupedActions,
    QConfig,
    QGreedyPolicy,
    StateEncoder,
    TabularQ,
    cem_search,
    evaluate_path,
    monte_carlo,
    sort_key,
    top_distinct,
    train_q_learning,
)

ANCHOR = "2017-06-01"


# --- Grouped actions --------------------------------------------------------------------------

def test_grouped_actions_mapping_and_bounds() -> None:
    acts = GroupedActions(0.15)
    assert len(acts) == 9 and acts.actions.shape == (9, 4)
    levels = {-0.30, -0.15, 0.0}
    for a in acts.actions:
        assert a[0] == a[1] and a[2] == a[3]                          # SPY = QQQ, XLF = XLE
        assert {round(float(a[0]), 6), round(float(a[2]), 6)} <= levels
        assert np.all(np.abs(a) <= 1.0)
    rows = {tuple(round(float(v), 6) for v in a) for a in acts.actions}
    assert (0.0, 0.0, 0.0, 0.0) in rows and (-0.15,) * 4 in rows and (-0.3,) * 4 in rows   # zero, scripted, 2L
    assert len(rows) == 9
    assert acts.label(0, ["SPY", "QQQ", "XLF", "XLE"]) == "SPY/QQQ -0.30, XLF/XLE -0.30"


def test_grouped_actions_reject_out_of_bound_levels_and_bad_groups() -> None:
    with pytest.raises(ValueError):
        GroupedActions(0.6)                                             # -2L = -1.2 < -1
    with pytest.raises(ValueError):
        GroupedActions(0.15, groups=((0, 1), (1, 2)))


# --- Tabular Q-learning -----------------------------------------------------------------------

def test_q_table_zero_initialised() -> None:
    q = TabularQ((2, 3), 4, gamma=0.9, alpha_min=0.05, bonus=1.0)
    assert q.Q.shape == (2, 3, 4) and not q.Q.any() and not q.N.any()


def test_terminal_updates_are_sample_means_with_visit_count_rate() -> None:
    q = TabularQ((1,), 2, gamma=0.9, alpha_min=0.05, bonus=1.0)
    s = (0,)
    q.update(s, 1, 1.0, s, terminal=True)                               # alpha = 1/1 -> Q = 1
    assert q.Q[s][1] == pytest.approx(1.0) and q.N[s][1] == 1
    q.update(s, 1, 3.0, s, terminal=True)                               # alpha = 1/2 -> Q = 2 (mean of 1, 3)
    assert q.Q[s][1] == pytest.approx(2.0) and q.N[s][1] == 2
    for _ in range(98):
        q.update(s, 1, 2.0, s, terminal=True)
    assert q.N[s][1] == 100                                             # alpha floor: max(0.05, 1/100) = 0.05


def test_nonterminal_update_bootstraps_from_visited_actions_only() -> None:
    q = TabularQ((2,), 2, gamma=0.9, alpha_min=0.05, bonus=1.0)
    s, s2 = (0,), (1,)
    q.Q[s2] = [5.0, -1.0]
    q.N[s2] = [0, 3]                                                    # action 0 never visited: its 5.0 is ignored
    td = q.update(s, 0, 1.0, s2, terminal=False)
    assert q.Q[s][0] == pytest.approx(1.0 + 0.9 * -1.0) and td == pytest.approx(0.1)
    q2 = TabularQ((2,), 2, gamma=0.9, alpha_min=0.05, bonus=1.0)
    q2.update(s, 0, 1.0, s2, terminal=False)                            # nothing visited in s2 -> value 0
    assert q2.Q[s][0] == pytest.approx(1.0)


def test_greedy_policy_never_prefers_unvisited_actions() -> None:
    q = TabularQ((1,), 3, gamma=0.9, alpha_min=0.05, bonus=1.0)
    s = (0,)
    assert q.greedy_action(s) is None                                   # nothing visited
    q.update(s, 2, -3.0, s, terminal=True)                              # only action 2 visited, negative value
    assert q.greedy_action(s) == 2                                      # unvisited zeros do not win
    q.update(s, 0, -1.0, s, terminal=True)
    assert q.greedy_action(s) == 0


def test_exploration_bonus_prefers_less_visited_when_values_tie() -> None:
    q = TabularQ((1,), 3, gamma=0.9, alpha_min=0.05, bonus=1.0)
    s = (0,)
    q.N[s] = [10, 0, 10]
    assert q.explore_action(s, eps=0.0, rng=np.random.default_rng(0)) == 1


def test_state_encoder_bins() -> None:
    idx = {"remaining_fraction": 0, "path_drawdown": 1, "path_historical_percentile": 2}
    enc = StateEncoder(idx, 5, dd_edges=[0.02, 0.04, 0.06, 0.08, 0.10], rank_edges=[0.01, 0.05, 0.20])
    assert enc.shape == (5, 6, 4)
    assert enc.encode(np.array([1.0, 0.0, 1.0])) == (0, 0, 3)
    assert enc.encode(np.array([0.4, 0.05, 0.03])) == (3, 2, 1)
    assert enc.encode(np.array([0.2, 0.10, 0.005])) == (4, 5, 0)       # edges are right-closed bins


# --- CEM logic (fake evaluator: no simulator) -------------------------------------------------

def _fake_evaluator(target: np.ndarray):
    """Valid iff the first coordinate is <= -0.5; rank rises as the path approaches ``target``."""
    calls: list[np.ndarray] = []

    def evaluate(x):
        x = np.asarray(x, dtype=float)
        calls.append(x.copy())
        valid = bool(x[0] <= -0.5)
        dist = float(np.linalg.norm(x - target))
        return {"A": x.reshape(1, -1), "days": 1, "invalid": False, "breached": valid, "valid": valid,
                "implausible": False, "rank": 1.0 / (1.0 + dist) if valid else 0.0, "log_tail": -dist,
                "L_star": -1.0, "drawdown": 0.11 if valid else 0.05, "dd_limit": 0.10}
    return evaluate, calls


def test_cem_budget_seed_paths_bounds_and_reproducibility() -> None:
    cfg = CEMConfig(restarts=3, pop=20, iters_per_restart=4, patience=10)
    target = np.array([-0.8, 0.2, -0.1, 0.0])
    seeds = [np.zeros(4), np.full(4, -0.15)]
    ev, calls = _fake_evaluator(target)
    out = cem_search(ev, 4, cfg, seed=7, seed_paths=seeds)
    assert out["evaluations"] == len(calls) <= cfg.budget == 240
    assert out["evaluations"] == 240                                    # no early stop with patience 10
    firsts = [i * 80 for i in range(3)]                                 # first batch of every restart
    for f in firsts:
        np.testing.assert_array_equal(calls[f], seeds[0])
        np.testing.assert_array_equal(calls[f + 1], seeds[1])
    assert all(np.abs(c).max() <= 1.0 for c in calls)
    assert out["best"]["valid"] and out["best"] is min(out["restart_bests"], key=sort_key)
    assert set(out["history"]["restart"]) == {0, 1, 2}
    seeded = [r for r in out["records"] if r["seeded"]]
    assert len(seeded) == 6 and all(r["iteration"] == 1 for r in seeded)          # 2 seed paths x 3 restarts
    assert not out["best_generated"]["seeded"]
    ev2, _ = _fake_evaluator(target)
    again = cem_search(ev2, 4, cfg, seed=7, seed_paths=seeds)
    assert again["best"]["rank"] == out["best"]["rank"]


def test_cem_patience_stops_only_after_a_valid_candidate() -> None:
    def flat_valid(x):
        return {"A": np.asarray(x).reshape(1, -1), "days": 1, "invalid": False, "breached": True, "valid": True,
                "implausible": False, "rank": 0.5, "log_tail": -1.0, "L_star": -2.0, "drawdown": 0.11, "dd_limit": 0.10}
    cfg = CEMConfig(restarts=1, pop=10, iters_per_restart=50, patience=3, sigma_floor=0.001)
    out = cem_search(flat_valid, 4, cfg, seed=0)
    assert len(out["history"]) == 4 and "gained" in out["stops"][0]     # gain at it 1, then 3 stale iterations

    def never_valid(x):
        return {"A": np.asarray(x).reshape(1, -1), "days": 1, "invalid": False, "breached": False, "valid": False,
                "implausible": False, "rank": 0.5, "log_tail": -1.0, "L_star": -2.0, "drawdown": 0.05, "dd_limit": 0.10}
    out = cem_search(never_valid, 4, CEMConfig(restarts=1, pop=10, iters_per_restart=8, patience=2, sigma_floor=0.001), seed=0)
    assert len(out["history"]) == 8                                     # no valid candidate -> patience never fires


def test_cem_config_validation() -> None:
    with pytest.raises(ValueError):
        CEMConfig(elite_frac=0.0)
    with pytest.raises(ValueError):
        CEMConfig(sigma_floor=0.6)
    with pytest.raises(ValueError):
        cem_search(lambda x: None, 4, CEMConfig(restarts=1, pop=4, iters_per_restart=1), seed=0, seed_paths=[np.full(4, 2.0)])


# --- Evaluation on the synthetic Stage 4 environment -------------------------------------------

def _mean_env(s4_source, s4_config):
    cfg = dataclasses.replace(s4_config, simulation=dataclasses.replace(s4_config.simulation, residual_mode="mean"))
    return make_env(cfg, s4_source)


def test_evaluate_path_stops_at_termination_and_ranks_feasibility_first(s4_source, s4_config) -> None:
    env = _mean_env(s4_source, s4_config)
    crash = evaluate_path(env, np.full((5, 4), -1.0), ANCHOR, seed=0)
    assert crash["breached"] and crash["days"] < 5 and len(crash["factor_shocks"]) == crash["days"]
    calm = evaluate_path(env, np.zeros((5, 4)), ANCHOR, seed=0)
    assert not calm["breached"] and calm["days"] == 5 and calm["rank"] == pytest.approx(1.0)
    valid = None
    for c in np.arange(0.05, 1.0, 0.025):
        r = evaluate_path(env, np.full((5, 4), -c), ANCHOR, seed=0)
        if r["breached"]:
            valid = r
            break
    assert valid is not None and valid["valid"]
    assert sort_key(valid) < sort_key(calm) and sort_key(valid) < sort_key(crash)
    with pytest.raises(ValueError):
        evaluate_path(env, np.full((5, 4), 1.5), ANCHOR, seed=0)
    assert top_distinct([calm, valid, crash], 5) == [valid]


def test_monte_carlo_budget_and_q_learning_smoke(s4_source, s4_config) -> None:
    env = _mean_env(s4_source, s4_config)
    mc = monte_carlo(lambda x: evaluate_path(env, x, ANCHOR, 0), 20, budget=15, seed=1)
    assert mc["evaluations"] == 15 and all(np.abs(r["A"]).max() <= 1.0 for r in mc["records"])
    obs_index = {n: i for i, n in enumerate(env.observation_names)}
    enc = StateEncoder(obs_index, env.horizon, [0.02, 0.04, 0.06, 0.08, 0.10], [0.01, 0.05, 0.20])
    acts = GroupedActions(0.15)
    cfg = QConfig(episodes=25)
    a1, log1 = train_q_learning(env, enc, acts, cfg, seed=3)
    a2, log2 = train_q_learning(env, enc, acts, cfg, seed=3)
    np.testing.assert_array_equal(a1.Q, a2.Q)                           # reproducible with a fixed seed
    assert int(a1.N.sum()) == int(log1["steps"].sum())                  # one visit per environment step
    pol = QGreedyPolicy(a1, enc, acts)
    obs, _ = env.reset(seed=0, options={"anchor_date": ANCHOR})
    a = pol.act(obs)
    assert a.shape == (4,) and np.abs(a).max() <= 1.0 and pol.steps == 1


# --- Regression: seeded reference paths must not stop or steer CEM (v2 full-run defect) -------

def _seed_only_valid(seed_path: np.ndarray):
    """Only the exact seed path is valid (rank 0.9); generated candidates are infeasible, closer as x[0] -> -1."""
    def evaluate(x):
        x = np.asarray(x, dtype=float)
        valid = bool(np.allclose(x, seed_path))
        return {"A": x.reshape(1, -1), "days": 1, "invalid": False, "breached": valid, "valid": valid,
                "implausible": False, "rank": 0.9 if valid else 0.0, "log_tail": -0.1 if valid else -5.0,
                "L_star": -1.0, "drawdown": 0.11 if valid else 0.05 - 0.04 * float(x[0]), "dd_limit": 0.10}
    return evaluate


def test_seeded_valid_reference_does_not_trigger_early_stop() -> None:
    seed_path = np.full(4, 0.9)
    cfg = CEMConfig(restarts=2, pop=20, iters_per_restart=8, patience=2, sigma_floor=0.001)
    out = cem_search(_seed_only_valid(seed_path), 4, cfg, seed=0, seed_paths=[seed_path])
    # v2 defect: the valid seed was each restart's best from iteration 1 -> patience stopped it at iteration 3
    assert len(out["history"]) == cfg.restarts * cfg.iters_per_restart
    assert out["evaluations"] == cfg.budget
    assert all(s == "iteration budget reached" for s in out["stops"])
    assert all(not b["seeded"] and not b["valid"] for b in out["restart_bests"])   # no CEM discovery claimed
    assert not out["best"]["valid"] and not out["best"]["seeded"]
    refs = out["references"]
    assert len(refs) == cfg.restarts and all(r["valid"] and r["seeded"] for r in refs)   # baselines still recorded
    assert not any(r["seeded"] and r["elite"] for r in out["records"])               # never steer the update
    assert out["history"]["restart best generated valid rank"].isna().all()


def test_patience_counts_from_the_first_generated_valid_candidate() -> None:
    def flat_valid(x):
        return {"A": np.asarray(x).reshape(1, -1), "days": 1, "invalid": False, "breached": True, "valid": True,
                "implausible": False, "rank": 0.5, "log_tail": -1.0, "L_star": -2.0, "drawdown": 0.11, "dd_limit": 0.10}

    def with_better_seed(x):
        rec = flat_valid(x)
        if np.allclose(x, 0.9):
            rec["rank"] = 0.95
        return rec
    cfg = CEMConfig(restarts=1, pop=10, iters_per_restart=50, patience=3, sigma_floor=0.001)
    out = cem_search(with_better_seed, 4, cfg, seed=0, seed_paths=[np.full(4, 0.9)])
    assert len(out["history"]) == 1 + cfg.patience                     # first generated valid at it 1, then 3 stale
    assert out["best"]["rank"] == 0.5 and not out["best"]["seeded"]    # the better seed is not the CEM result


def test_seed_paths_must_leave_room_for_generated_candidates() -> None:
    with pytest.raises(ValueError):
        cem_search(lambda x: None, 4, CEMConfig(restarts=1, pop=2, iters_per_restart=1), seed=0,
                   seed_paths=[np.zeros(4), np.zeros(4)])


# --- Experiment 0 / Experiment 1 -------------------------------------------------------------

from types import SimpleNamespace

from rlstress.search import (
    enumerate_sequences,
    episode_metrics,
    margin_bonus_for,
    monte_carlo_sequences,
    plausibility_margin,
)


def test_plausibility_margin_values_and_bounds() -> None:
    q = 0.01
    assert plausibility_margin(q, q) == 0.0 and plausibility_margin(0.005, q) == 0.0
    assert plausibility_margin(1.0, q) == pytest.approx(1.0) and plausibility_margin(1.5, q) == 1.0
    assert plausibility_margin(0.0808, q) == pytest.approx(np.log(8.08) / np.log(100))          # ~0.454
    assert plausibility_margin(0.0205, q) == pytest.approx(np.log(2.05) / np.log(100))          # ~0.156
    assert plausibility_margin(float("nan"), q) == 0.0 and plausibility_margin(None, q) == 0.0
    xs = np.linspace(0.0, 1.0, 101)
    vals = [plausibility_margin(x, q) for x in xs]
    assert all(a <= b for a, b in zip(vals, vals[1:]))


def test_margin_bonus_only_where_valid_breach_reward_is_paid() -> None:
    valid = {"valid_breach": True, "path_historical_percentile": 0.0808, "reward_components": {"breach": 10.0}}
    assert margin_bonus_for(valid, 5.0, 0.01) == pytest.approx(5.0 * plausibility_margin(0.0808, 0.01))
    assert margin_bonus_for(valid, 0.0, 0.01) == 0.0
    implausible = {"valid_breach": False, "implausible_breach": True, "path_historical_percentile": 0.002, "reward_components": {"breach": 0.0}}
    assert margin_bonus_for(implausible, 5.0, 0.01) == 0.0
    nonterminal = {"valid_breach": False, "path_historical_percentile": 0.5, "reward_components": {"breach": 0.0}}
    assert margin_bonus_for(nonterminal, 5.0, 0.01) == 0.0
    already_paid = {"valid_breach": True, "path_historical_percentile": 0.08, "reward_components": {"breach": 0.0}}
    assert margin_bonus_for(already_paid, 5.0, 0.01) == 0.0                                       # no duplicate
    with pytest.raises(ValueError):
        QConfig(margin_bonus=-1.0)


def _q_setup(s4_source, s4_config, sample=False):
    cfg = s4_config if sample else dataclasses.replace(s4_config, simulation=dataclasses.replace(s4_config.simulation, residual_mode="mean"))
    env = make_env(cfg, s4_source)
    obs_index = {n: i for i, n in enumerate(env.observation_names)}
    enc = StateEncoder(obs_index, env.horizon, [0.02, 0.04, 0.06, 0.08, 0.10], [0.01, 0.05, 0.20])
    return env, enc, GroupedActions(0.15)


def test_beta_zero_reproduces_unchanged_training(s4_source, s4_config) -> None:
    env, enc, acts = _q_setup(s4_source, s4_config, sample=True)
    a, log_a = train_q_learning(env, enc, acts, QConfig(episodes=60), seed=43)
    b, log_b = train_q_learning(env, enc, acts, QConfig(episodes=60, margin_bonus=0.0), seed=43)
    np.testing.assert_array_equal(a.Q, b.Q)
    np.testing.assert_array_equal(a.N, b.N)
    assert log_a.equals(log_b) and (log_a["margin_bonus"] == 0).all()


def test_beta_bonus_applied_once_on_valid_terminal_transitions(s4_source, s4_config) -> None:
    env, enc, acts = _q_setup(s4_source, s4_config, sample=True)
    beta, q = 5.0, env.sim_config.min_plausibility
    _, log = train_q_learning(env, enc, acts, QConfig(episodes=150, margin_bonus=beta), seed=7)
    assert log["valid"].any() and (~log["valid"]).any()
    for _, row in log.iterrows():
        expected = beta * plausibility_margin(row["final_rank"], q) if row["valid"] else 0.0
        assert row["margin_bonus"] == pytest.approx(expected)                                       # one bonus per valid episode
    assert (log.loc[log["implausible"], "margin_bonus"] == 0).all()


def test_enumeration_count_bounds_and_prefix_reuse_match_brute_force(s4_source, s4_config) -> None:
    env, _, _ = _q_setup(s4_source, s4_config)
    small = GroupedActions(0.15, multipliers=(-2.0, 0.0))                    # 4 actions -> 4**5 = 1024 sequences
    evaluate = lambda x: evaluate_path(env, x, ANCHOR, 0)
    df = enumerate_sequences(evaluate, small, env.horizon)
    assert len(df) == 4 ** 5 and df["sequence"].nunique() == 4 ** 5
    assert all(np.abs(small.actions[list(s)]).max() <= 1.0 for s in df["sequence"])
    assert df.attrs["simulations"] < len(df)                                  # reuse happened (early breaches)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(df), 60, replace=False):                         # reused rows equal direct evaluation
        direct = evaluate(small.actions[list(df.loc[i, "sequence"])])
        for k in ("days", "breached", "valid", "rank", "drawdown", "S", "mean_sq_action"):
            assert df.loc[i, k] == pytest.approx(direct[k]), k


def test_episode_metrics_denominators() -> None:
    rec = lambda v, b, imp, rank, dd, day, a2: SimpleNamespace(valid_breach=v, breached=b, implausible_breach=imp,
                                                               historical_percentile=rank, path_drawdown=dd, breach_day=day, mean_sq_action=a2)
    recs = [rec(True, True, False, 0.05, 0.11, 3, 0.04), rec(True, True, False, 0.02, 0.12, 2, 0.06),
            rec(False, True, True, 0.001, 0.15, 1, 0.3), rec(False, False, False, 0.9, 0.03, None, 0.0)]
    m = episode_metrics(recs)
    assert m["episodes"] == 4 and m["valid rate"] == 0.5 and m["any-breach rate"] == 0.75 and m["implausible rate"] == 0.25
    assert m["median valid rank"] == pytest.approx(0.035) and m["p10 valid rank"] == pytest.approx(0.023)
    assert m["mean breach day"] == pytest.approx(2.0)


def test_monte_carlo_sequences_budget_and_common_seeds(s4_source, s4_config) -> None:
    env, _, acts = _q_setup(s4_source, s4_config, sample=True)
    out = monte_carlo_sequences(env, acts, step_budget=200, draws=4, draw_seed=50_000, seed=1)
    df = out["candidates"]
    assert out["steps"] >= 200 and out["steps"] - 4 * env.horizon < 200     # stops at the first candidate past the budget
    assert all(len(s) == env.horizon and set(s) <= set(range(len(acts))) for s in df["sequence"])
    assert np.abs(out["selected path"]).max() <= 1.0
    again = monte_carlo_sequences(env, acts, step_budget=200, draws=4, draw_seed=50_000, seed=1)
    assert again["candidates"].equals(df)
