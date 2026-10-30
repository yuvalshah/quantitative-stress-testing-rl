"""Stage 5 search over complete shock paths: evaluator, Monte Carlo, restarted CEM and tabular Q-learning.

Every candidate is executed by the Stage 4 :class:`~rlstress.env.StressTestEnv`; this module adds no
financial logic. A *valid* candidate breaches a path limit and passes the plausibility gate
(``valid_breach`` from the environment). Candidates are ranked feasibility-first: valid ones by
percentile rank (higher = more plausible), then log-tail score; the rest by their distance to
feasibility. No penalty is added to make an infeasible candidate look feasible.

Q-learning works on a small grouped action set (equity block SPY/QQQ, sector block XLF/XLE) and a
binned state read from the observation. Its table starts at zero; exploration (epsilon-greedy plus a
visit-count bonus) is used in training only, and the greedy policy chooses among *visited* actions so
an untried action can never win merely because of its initial value.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from rlstress.env import StressTestEnv


# --- Candidate evaluation --------------------------------------------------------------------

def drawdown_limit(env: StressTestEnv) -> float:
    """Threshold of the environment's 5-day drawdown limit."""
    return next(l.threshold for l in env.path_limits if l.metric == "max_drawdown")


def evaluate_path(env: StressTestEnv, path: np.ndarray, anchor: Any, seed: int) -> dict[str, Any]:
    """Run an ``(horizon, k)`` action matrix through ``env`` and stop where the environment stops.

    Days after termination are not executed (``days`` < horizon); they are never treated as zero shocks.
    """
    h, k = env.horizon, env.simulator.model.k
    A = np.asarray(path, dtype=float).reshape(h, k)
    if not np.all(np.isfinite(A)) or np.any(np.abs(A) > 1.0):
        raise ValueError("candidate outside [-1, 1] or not finite")
    env.reset(seed=seed, options={"anchor_date": anchor})
    infos, total = [], 0.0
    for d in range(h):
        _, r, term, _, info = env.step(A[d].astype(np.float32))
        total += r
        infos.append(info)
        if term:
            break
    last, limit = infos[-1], drawdown_limit(env)
    if last.get("invalid"):
        return {"A": A, "days": len(infos) - 1, "invalid": True, "breached": False, "valid": False,
                "implausible": False, "rank": 0.0, "log_tail": -np.inf, "L_star": np.nan, "S": np.inf,
                "drawdown": np.nan, "loss": np.nan, "return": total, "mean_sq_action": float(np.mean(A ** 2)),
                "dd_limit": limit, "values": [], "factor_shocks": [], "stock_returns": []}
    t = last["step"]
    return {"A": A, "days": t, "invalid": False, "breached": last["breached"], "valid": last["valid_breach"],
            "implausible": last["implausible_breach"], "rank": last["path_historical_percentile"],
            "log_tail": last["path_log_tail_probability"], "L_star": env.log_thresholds[t],
            "S": float(sum(i["plausibility"]["surprisal"] for i in infos)), "drawdown": last["path_drawdown"],
            "loss": last["cumulative_loss"], "return": total, "mean_sq_action": float(np.mean(A[:t] ** 2)),
            "dd_limit": limit, "values": [env.portfolio.initial_capital] + [i["portfolio_value"] for i in infos],
            "factor_shocks": [list(i["factor_shock"].values()) for i in infos],
            "stock_returns": [list(i["stock_returns"].values()) for i in infos]}


def violation(rec: dict[str, Any]) -> float:
    """0 for a valid candidate; otherwise drawdown shortfall plus plausibility gap (ranking only)."""
    if rec["valid"]:
        return 0.0
    if rec["invalid"]:
        return float("inf")
    v = max(0.0, rec["dd_limit"] - rec["drawdown"]) / rec["dd_limit"]
    if rec["log_tail"] < rec["L_star"]:
        v += (rec["L_star"] - rec["log_tail"]) / abs(rec["L_star"]) if np.isfinite(rec["log_tail"]) else float("inf")
    return max(v, 1e-12)


def sort_key(rec: dict[str, Any]) -> tuple:
    """Feasibility first; valid candidates by higher rank, then higher log-tail score."""
    return (0, -rec["rank"], -rec["log_tail"]) if rec["valid"] else (1, violation(rec), -rec["rank"])


def best_so_far(records: Sequence[dict[str, Any]]) -> np.ndarray:
    """Best valid rank after each evaluation (NaN until the first valid candidate)."""
    out, best = [], None
    for r in records:
        if r["valid"] and (best is None or sort_key(r) < sort_key(best)):
            best = r
        out.append(best["rank"] if best is not None else np.nan)
    return np.array(out)


# --- Monte Carlo and CEM -----------------------------------------------------------------------

def monte_carlo(evaluate: Callable[[np.ndarray], dict], dim: int, budget: int, seed: int) -> dict[str, Any]:
    """Uniform paths on [-1, 1]^dim, ``budget`` evaluations."""
    X = np.random.default_rng(seed).uniform(-1.0, 1.0, size=(budget, dim))
    recs = [evaluate(x) for x in X]
    return {"records": recs, "best": min(recs, key=sort_key), "evaluations": len(recs)}


@dataclass(frozen=True)
class CEMConfig:
    """Restarted CEM. Total budget = restarts * pop * iters_per_restart candidate evaluations (an upper bound)."""

    restarts: int = 5
    pop: int = 100
    iters_per_restart: int = 12
    elite_frac: float = 0.20
    smoothing: float = 0.40
    sigma0: float = 0.5
    sigma_floor: float = 0.02
    patience: int = 6
    min_rank_gain: float = 0.002

    @property
    def budget(self) -> int:
        """Maximum number of candidate evaluations."""
        return self.restarts * self.pop * self.iters_per_restart

    def __post_init__(self) -> None:
        if min(self.restarts, self.pop, self.iters_per_restart, self.patience) < 1:
            raise ValueError("restarts, pop, iters_per_restart and patience must be >= 1")
        if not 0.0 < self.elite_frac <= 1.0 or not 0.0 < self.smoothing <= 1.0:
            raise ValueError("elite_frac and smoothing must be in (0, 1]")
        if self.sigma0 <= 0 or not 0.0 < self.sigma_floor < self.sigma0 or self.min_rank_gain < 0:
            raise ValueError("need sigma0 > sigma_floor > 0 and min_rank_gain >= 0")


def cem_search(evaluate: Callable[[np.ndarray], dict], dim: int, cfg: CEMConfig, seed: int,
               seed_paths: Sequence[np.ndarray] = ()) -> dict[str, Any]:
    """Independent CEM restarts with a shared evaluator.

    Each restart samples ``N(mu, diag sigma^2)`` projected onto [-1, 1]. Optional ``seed_paths`` (e.g. the
    zero and scripted baselines) replace the first candidates of every restart's first batch as
    *reference* evaluations: they are recorded (``seeded=True``) and count towards the budget, but they
    never enter the elite set, the restart's best-so-far, the valid-candidate progress or the patience
    counter, so a reference path can neither steer nor stop the search nor be reported as a discovery.

    Elites are the top ``elite_frac`` of the batch's *generated* candidates by :func:`sort_key`; ``mu`` and
    ``sigma`` move towards the elite mean/std with ``smoothing``, ``sigma >= sigma_floor``. A restart stops
    after ``iters_per_restart`` iterations, or when its best *generated* valid rank has not risen by at least
    ``min_rank_gain`` for ``patience`` iterations (counted only after the restart has generated a valid
    candidate itself), or when ``sigma`` has collapsed to the floor.
    """
    seeds = [np.asarray(p, dtype=float).ravel() for p in seed_paths]
    if any(p.shape != (dim,) or np.abs(p).max() > 1.0 for p in seeds):
        raise ValueError("seed paths must have the candidate dimension and lie in [-1, 1]")
    if len(seeds) >= cfg.pop:
        raise ValueError("seed paths must leave room for generated candidates in the first batch")
    records, history, restart_bests, stops = [], [], [], []
    for r in range(cfg.restarts):
        rng = np.random.default_rng([seed, r])
        mu, sigma = np.zeros(dim), np.full(dim, cfg.sigma0)
        best, best_rank_at_gain, stale, stop = None, -np.inf, 0, "iteration budget reached"
        for it in range(1, cfg.iters_per_restart + 1):
            X = np.clip(mu + sigma * rng.standard_normal((cfg.pop, dim)), -1.0, 1.0)
            n_seed = len(seeds) if it == 1 else 0
            if n_seed:
                X[:n_seed] = np.vstack(seeds)
            batch = [evaluate(x) for x in X]
            for i, rec in enumerate(batch):
                rec.update(restart=r, iteration=it, seeded=i < n_seed, elite=False)
            records.extend(batch)
            gen_idx = list(range(n_seed, cfg.pop))                        # generated candidates only
            n_elite = max(2, int(round(cfg.elite_frac * len(gen_idx))))
            order = sorted(gen_idx, key=lambda i: sort_key(batch[i]))
            elites = [batch[i] for i in order[:n_elite]]
            for e in elites:
                e["elite"] = True
            E = X[order[:n_elite]]
            mu = (1 - cfg.smoothing) * mu + cfg.smoothing * E.mean(axis=0)
            sigma = np.maximum((1 - cfg.smoothing) * sigma + cfg.smoothing * E.std(axis=0), cfg.sigma_floor)
            if best is None or sort_key(batch[order[0]]) < sort_key(best):
                best = batch[order[0]]
            if best["valid"]:
                if best["rank"] >= best_rank_at_gain + cfg.min_rank_gain:
                    best_rank_at_gain, stale = best["rank"], 0
                else:
                    stale += 1
            feas_elites = [e["rank"] for e in elites if e["valid"]]
            diffs = E[:, None, :] - E[None, :, :]
            history.append({"restart": r, "iteration": it, "evaluations": len(records),
                            "valid generated in batch": sum(batch[i]["valid"] for i in gen_idx),
                            "restart best generated valid rank": best["rank"] if best["valid"] else np.nan,
                            "elite valid share": len(feas_elites) / n_elite,
                            "elite mean valid rank": float(np.mean(feas_elites)) if feas_elites else np.nan,
                            "mean sigma": float(sigma.mean()),
                            "elite mean pairwise distance": float(np.sqrt((diffs ** 2).sum(-1))[np.triu_indices(n_elite, 1)].mean())})
            if best["valid"] and stale >= cfg.patience:
                stop = f"best generated valid rank gained < {cfg.min_rank_gain} for {cfg.patience} iterations"
                break
            if sigma.mean() <= cfg.sigma_floor * 1.05:
                stop = "sampling distribution collapsed to the sigma floor"
                break
        restart_bests.append(best)
        stops.append(stop)
    best = min(restart_bests, key=sort_key)
    return {"records": records, "history": pd.DataFrame(history), "restart_bests": restart_bests,
            "stops": stops, "best": best, "best_generated": best, "evaluations": len(records),
            "references": [rec for rec in records if rec["seeded"]]}


def top_distinct(records: Sequence[dict[str, Any]], n: int, min_distance: float = 0.05) -> list[dict[str, Any]]:
    """Up to ``n`` valid candidates, best first, at least ``min_distance`` apart on their executed days."""
    out: list[dict[str, Any]] = []
    for rec in sorted((r for r in records if r["valid"]), key=sort_key):
        if all(rec["days"] != o["days"] or np.linalg.norm(rec["A"][: rec["days"]] - o["A"][: o["days"]]) >= min_distance
               for o in out):
            out.append(rec)
        if len(out) == n:
            break
    return out


# --- Policies used with rlstress.rl.evaluation.evaluate_policy ---------------------------------

def day_index(obs: np.ndarray, obs_index: dict[str, int], horizon: int) -> int:
    """Current day (0-based) from the observation's remaining fraction."""
    return min(horizon - 1, int(round((1.0 - float(obs[obs_index["remaining_fraction"]])) * horizon)))


class OpenLoopPolicy:
    """Replays a fixed ``(horizon, k)`` path."""

    def __init__(self, path: np.ndarray, obs_index: dict[str, int], name: str) -> None:
        self.path = np.asarray(path, dtype=np.float32)
        self.obs_index, self.name = obs_index, name

    def reset(self, seed: int) -> None:
        """Stateless."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Action of the current day."""
        return self.path[day_index(obs, self.obs_index, len(self.path))]


# --- Tabular Q-learning --------------------------------------------------------------------------

class GroupedActions:
    """Discrete joint actions: one level per factor group, copied to every factor in the group.

    Default groups: (SPY, QQQ) and (XLF, XLE), levels ``multipliers * level`` in Stage 4 action units,
    so each factor keeps its own scale ``a * b_j`` in the environment.
    """

    def __init__(self, level: float, k: int = 4, groups: Sequence[Sequence[int]] = ((0, 1), (2, 3)),
                 multipliers: Sequence[float] = (-2.0, -1.0, 0.0)) -> None:
        if sorted(i for g in groups for i in g) != list(range(k)):
            raise ValueError("groups must partition the factor indices")
        self.levels = [m * level for m in multipliers]
        if any(abs(v) > 1.0 for v in self.levels):
            raise ValueError(f"levels {self.levels} exceed the action bound 1")
        self.groups = [tuple(g) for g in groups]
        self.combos = list(itertools.product(range(len(self.levels)), repeat=len(self.groups)))
        acts = []
        for combo in self.combos:
            a = np.zeros(k, dtype=np.float32)
            for g, li in zip(self.groups, combo):
                a[list(g)] = self.levels[li]
            acts.append(a)
        self.actions = np.vstack(acts)

    def __len__(self) -> int:
        return len(self.actions)

    def label(self, i: int, names: Sequence[str]) -> str:
        """Readable description of action ``i``."""
        return ", ".join(f"{'/'.join(names[j] for j in g)} {self.levels[li]:+.2f}" for g, li in zip(self.groups, self.combos[i]))


class StateEncoder:
    """Discrete state (day, drawdown bin, rank bin) read from the Stage 4 observation."""

    def __init__(self, obs_index: dict[str, int], horizon: int, dd_edges: Sequence[float], rank_edges: Sequence[float]) -> None:
        self.obs_index, self.horizon = obs_index, horizon
        self.dd_edges, self.rank_edges = np.asarray(dd_edges, float), np.asarray(rank_edges, float)
        self.shape = (horizon, len(self.dd_edges) + 1, len(self.rank_edges) + 1)

    def encode(self, obs: np.ndarray) -> tuple[int, int, int]:
        """State tuple for an observation."""
        dd = int(np.searchsorted(self.dd_edges, float(obs[self.obs_index["path_drawdown"]]), side="right"))
        rk = int(np.searchsorted(self.rank_edges, float(obs[self.obs_index["path_historical_percentile"]]), side="right"))
        return day_index(obs, self.obs_index, self.horizon), dd, rk


@dataclass(frozen=True)
class QConfig:
    """Q-learning settings. ``alpha = max(alpha_min, 1/n)`` with n the visits of (s, a).

    ``margin_bonus`` (beta, default 0 = unchanged Stage 4 reward): on the single transition where the
    environment pays the valid-breach reward, the learning target also receives
    ``beta * plausibility_margin(rank, q)``. Implausible breaches and non-terminal steps never receive it.
    """

    episodes: int = 20_000
    gamma: float = 0.99
    alpha_min: float = 0.05
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_share: float = 0.8
    bonus: float = 1.0
    margin_bonus: float = 0.0

    def __post_init__(self) -> None:
        if self.margin_bonus < 0:
            raise ValueError("margin_bonus must be >= 0")


def plausibility_margin(rank: float | None, q: float) -> float:
    """``ln(rank / q) / ln(1 / q)`` clamped to [0, 1]; 0 when rank <= q or the rank is missing/not finite."""
    if rank is None or not np.isfinite(rank) or rank <= q:
        return 0.0
    return float(np.clip(np.log(rank / q) / np.log(1.0 / q), 0.0, 1.0))


def margin_bonus_for(info: dict[str, Any], beta: float, q: float) -> float:
    """Bonus for one transition: only where the environment pays the valid-breach reward (once per episode)."""
    if beta == 0.0 or not info.get("valid_breach") or info.get("reward_components", {}).get("breach", 0.0) <= 0.0:
        return 0.0
    return beta * plausibility_margin(info.get("path_historical_percentile"), q)


class TabularQ:
    """Zero-initialised Q-table with visit counts."""

    def __init__(self, state_shape: Sequence[int], n_actions: int, gamma: float, alpha_min: float, bonus: float) -> None:
        self.Q = np.zeros((*state_shape, n_actions))
        self.N = np.zeros((*state_shape, n_actions), dtype=np.int64)
        self.gamma, self.alpha_min, self.bonus = gamma, alpha_min, bonus

    def update(self, s: tuple, a: int, r: float, s_next: tuple, terminal: bool) -> float:
        """``Q += alpha (r + gamma max Q(s') - Q)``, no bootstrap from terminal states; returns the TD error."""
        self.N[s][a] += 1
        alpha = max(self.alpha_min, 1.0 / self.N[s][a])
        target = r if terminal else r + self.gamma * self.greedy_value(s_next)
        td = target - self.Q[s][a]
        self.Q[s][a] += alpha * td
        return float(td)

    def greedy_value(self, s: tuple) -> float:
        """max over visited actions of Q(s, .); 0 if nothing in ``s`` was visited (the initial value)."""
        visited = self.N[s] > 0
        return float(self.Q[s][visited].max()) if visited.any() else 0.0

    def explore_action(self, s: tuple, eps: float, rng: np.random.Generator) -> int:
        """Training only: epsilon-greedy on ``Q + bonus / sqrt(N + 1)``, random tie-break."""
        if rng.random() < eps:
            return int(rng.integers(self.Q.shape[-1]))
        score = self.Q[s] + self.bonus / np.sqrt(self.N[s] + 1.0)
        return int(rng.choice(np.flatnonzero(score == score.max())))

    def greedy_action(self, s: tuple) -> int | None:
        """Best *visited* action (lowest index on ties); ``None`` if no action was ever tried in ``s``."""
        visited = self.N[s] > 0
        if not visited.any():
            return None
        return int(np.argmax(np.where(visited, self.Q[s], -np.inf)))


def train_q_learning(env: StressTestEnv, encoder: StateEncoder, actions: GroupedActions, cfg: QConfig,
                     seed: int) -> tuple[TabularQ, pd.DataFrame]:
    """Epsilon-greedy Q-learning with a visit-count bonus; one environment reset per episode (seed + ep)."""
    rng = np.random.default_rng(seed)
    agent = TabularQ(encoder.shape, len(actions), cfg.gamma, cfg.alpha_min, cfg.bonus)
    decay = max(1, int(cfg.eps_decay_share * cfg.episodes))
    q = env.sim_config.min_plausibility
    log = []
    for ep in range(cfg.episodes):
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * ep / decay)
        obs, _ = env.reset(seed=seed + ep)
        s, total, bonus_total, steps, info = encoder.encode(obs), 0.0, 0.0, 0, {}
        while True:
            a = agent.explore_action(s, eps, rng)
            obs, r, term, trunc, info = env.step(actions.actions[a])
            s_next = encoder.encode(obs)
            bonus = margin_bonus_for(info, cfg.margin_bonus, q)
            agent.update(s, a, r + bonus, s_next, term or trunc)
            total, bonus_total, steps, s = total + r, bonus_total + bonus, steps + 1, s_next
            if term or trunc:
                break
        log.append({"episode": ep, "epsilon": eps, "return": total, "margin_bonus": bonus_total, "steps": steps,
                    "breached": bool(info.get("breached")), "valid": bool(info.get("valid_breach")),
                    "implausible": bool(info.get("implausible_breach")),
                    "final_rank": info.get("path_historical_percentile", np.nan)})
    return agent, pd.DataFrame(log)


class QGreedyPolicy:
    """Greedy policy over visited actions; falls back to the zero action (counted) in never-visited states."""

    name = "q_learning_greedy"

    def __init__(self, agent: TabularQ, encoder: StateEncoder, actions: GroupedActions) -> None:
        self.agent, self.encoder, self.actions = agent, encoder, actions
        self.zero = int(np.flatnonzero(np.all(actions.actions == 0, axis=1))[0])
        self.steps = self.fallbacks = 0

    def reset(self, seed: int) -> None:
        """Stateless between episodes."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Greedy action for the observation's state."""
        a = self.agent.greedy_action(self.encoder.encode(obs))
        self.steps += 1
        if a is None:
            self.fallbacks += 1
            a = self.zero
        return self.actions.actions[a]


# --- Experiment 0: exhaustive enumeration of the discrete action space --------------------------

ENUM_FIELDS = ("days", "breached", "valid", "implausible", "rank", "log_tail", "drawdown", "loss", "S", "mean_sq_action")


def enumerate_sequences(evaluate: Callable[[np.ndarray], dict], actions: GroupedActions, horizon: int) -> pd.DataFrame:
    """Every ``len(actions) ** horizon`` fixed sequence, evaluated with a *deterministic* evaluator.

    A sequence that terminates after ``t < horizon`` days executes only its first ``t`` actions, so every
    other sequence with the same first ``t`` actions has an identical outcome; such outcomes are reused
    instead of re-simulated. ``df.attrs["simulations"]`` counts the simulator runs actually performed.
    """
    rows, terminated, sims = [], {}, 0
    for seq in itertools.product(range(len(actions)), repeat=horizon):
        hit = next((terminated[seq[:t]] for t in range(1, horizon) if seq[:t] in terminated), None)
        if hit is None:
            rec = evaluate(actions.actions[list(seq)])
            sims += 1
            hit = {k: rec[k] for k in ENUM_FIELDS}
            if rec["days"] < horizon:
                terminated[seq[: rec["days"]]] = hit
        rows.append({"sequence": seq, **hit})
    df = pd.DataFrame(rows)
    df.attrs["simulations"] = sims
    return df


# --- Episode-level metrics and the discrete Monte Carlo variant -------------------------------

def episode_metrics(records: Sequence[Any]) -> dict[str, float]:
    """Rates and valid-episode plausibility from EpisodeRecord-like objects (denominator: episodes)."""
    n = len(records)
    valid = [r for r in records if r.valid_breach]
    breached = [r for r in records if r.breached]
    ranks = np.array([r.historical_percentile for r in valid], dtype=float)
    return {"episodes": n, "valid rate": len(valid) / n, "any-breach rate": len(breached) / n,
            "implausible rate": sum(r.implausible_breach for r in records) / n,
            "median valid rank": float(np.median(ranks)) if len(ranks) else np.nan,
            "p10 valid rank": float(np.quantile(ranks, 0.10)) if len(ranks) else np.nan,
            "mean drawdown at breach": float(np.mean([r.path_drawdown for r in breached])) if breached else np.nan,
            "mean breach day": float(np.mean([r.breach_day for r in breached])) if breached else np.nan,
            "mean a^2": float(np.mean([r.mean_sq_action for r in records]))}


def rollout_path(env: StressTestEnv, path: np.ndarray, seed: int) -> dict[str, Any]:
    """One open-loop episode of a fixed path (environment-drawn anchor and residuals for ``seed``)."""
    env.reset(seed=seed)
    info, steps = {}, 0
    for d in range(env.horizon):
        _, _, term, trunc, info = env.step(np.asarray(path[d], dtype=np.float32))
        steps += 1
        if term or trunc:
            break
    return {"steps": steps, "valid": bool(info.get("valid_breach")), "breached": bool(info.get("breached")),
            "rank": info.get("path_historical_percentile", np.nan)}


def monte_carlo_sequences(env: StressTestEnv, actions: GroupedActions, step_budget: int, draws: int,
                          draw_seed: int, seed: int, p_valid_target: float = 0.80) -> dict[str, Any]:
    """Random fixed sequences from the discrete action set, each scored on the same ``draws`` episodes.

    Candidates are drawn uniformly until ``step_budget`` environment steps are used (common random
    numbers: every candidate sees episode seeds ``draw_seed .. draw_seed + draws - 1``). Selection: among
    candidates with valid share >= ``p_valid_target`` the highest median valid rank; otherwise the highest
    valid share (then median valid rank).
    """
    rng = np.random.default_rng(seed)
    rows, steps = [], 0
    while steps < step_budget:
        idx = rng.integers(len(actions), size=env.horizon)
        path = actions.actions[idx]
        outs = [rollout_path(env, path, draw_seed + j) for j in range(draws)]
        steps += sum(o["steps"] for o in outs)
        ranks = [o["rank"] for o in outs if o["valid"]]
        rows.append({"sequence": tuple(int(i) for i in idx), "valid share": float(np.mean([o["valid"] for o in outs])),
                     "breach share": float(np.mean([o["breached"] for o in outs])),
                     "median valid rank": float(np.median(ranks)) if ranks else np.nan})
    df = pd.DataFrame(rows)
    ok = df[df["valid share"] >= p_valid_target]
    pick = (ok.sort_values("median valid rank", ascending=False) if len(ok)
            else df.sort_values(["valid share", "median valid rank"], ascending=False)).iloc[0]
    return {"candidates": df, "selected": pick, "steps": steps, "meets target": bool(len(ok)),
            "selected path": actions.actions[list(pick["sequence"])]}
