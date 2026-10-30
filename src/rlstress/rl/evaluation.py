"""Episode evaluation, metrics, scenario diversity and reward-hacking diagnostics.

Protocol: every policy is run on the same list of episode seeds; each episode
calls ``env.reset(seed=s)`` (same anchor date and residual RNG for every
policy: common random numbers) and ``policy.reset(s)``.

Termination reasons: ``valid_breach`` (breach with a path no more extreme than
the ``min_plausibility`` share of real training windows), ``implausible_breach``,
``horizon`` (no breach in 5 days) and ``invalid`` (simulator failure).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from rlstress.env import StressTestEnv
from rlstress.rl.policies import Policy

COMPONENTS = ("breach", "severity", "plausibility", "magnitude", "step", "invalid", "plausibility_credit")


@dataclass
class EpisodeRecord:
    """Outcome of one evaluation episode."""

    policy: str
    episode_seed: int
    anchor_date: str
    length: int
    total_reward: float
    components: dict[str, float]
    termination: str
    breached: bool
    valid_breach: bool
    implausible_breach: bool
    drawdown_breach: bool
    loss_breach: bool
    cumulative_loss: float
    path_drawdown: float
    breach_day: int | None
    historical_percentile: float
    path_log_tail_probability: float
    mean_abs_action: float
    mean_sq_action: float
    saturated_action_share: float
    residual_draws: float
    actions: list[list[float]] = field(default_factory=list)
    factor_shocks: list[list[float]] = field(default_factory=list)
    stock_returns: list[list[float]] = field(default_factory=list)
    value_path: list[float] = field(default_factory=list)

    def flat(self) -> dict[str, Any]:
        """One-row representation without the path arrays."""
        row = {k: v for k, v in asdict(self).items()
               if k not in ("actions", "factor_shocks", "stock_returns", "value_path", "components")}
        row.update({f"reward_{k}": self.components.get(k, 0.0) for k in COMPONENTS})
        return row


def run_episode(env: StressTestEnv, policy: Policy, seed: int) -> EpisodeRecord:
    """Run one episode and collect its outcome."""
    obs, info = env.reset(seed=seed)
    policy.reset(seed)
    comps = dict.fromkeys(COMPONENTS, 0.0)
    total, actions, draws = 0.0, [], []
    while True:
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        for k, v in info["reward_components"].items():
            comps[k] += v
        if "action" in info:
            actions.append(info["action"])
            draws.append(info["residual_draws"])
        if terminated or truncated:
            break
    invalid = bool(info.get("invalid"))
    limits = {x["metric"]: x for x in info.get("limit_results", [])}
    breached = bool(info.get("breached", False))
    if invalid:
        termination = "invalid"
    elif info.get("valid_breach"):
        termination = "valid_breach"
    elif info.get("implausible_breach"):
        termination = "implausible_breach"
    else:
        termination = "horizon"
    scenario = env.scenario()
    result = env.current_result()
    a = np.asarray(actions, dtype=float) if actions else np.zeros((0, env.simulator.model.k))
    return EpisodeRecord(
        policy=policy.name, episode_seed=seed, anchor_date=env.anchor.date().isoformat(),
        length=int(scenario.n_steps), total_reward=float(total), components=comps,
        termination=termination, breached=breached and not invalid,
        valid_breach=termination == "valid_breach", implausible_breach=termination == "implausible_breach",
        drawdown_breach=bool(limits.get("max_drawdown", {}).get("breached")),
        loss_breach=bool(limits.get("cumulative_loss", {}).get("breached")),
        cumulative_loss=float(result.loss_pct) if result else 0.0,
        path_drawdown=float(max(0.0, -result.max_drawdown)) if result else 0.0,
        breach_day=int(scenario.n_steps) if breached and not invalid else None,
        historical_percentile=float(info.get("path_historical_percentile", 1.0)),
        path_log_tail_probability=float(info.get("path_log_tail_probability", 0.0)),
        mean_abs_action=float(np.abs(a).mean()) if a.size else 0.0,
        mean_sq_action=float((a ** 2).mean()) if a.size else 0.0,
        saturated_action_share=float((np.abs(a) >= 0.99).mean()) if a.size else 0.0,
        residual_draws=float(np.mean(draws)) if draws else 0.0,
        actions=a.round(6).tolist(), factor_shocks=scenario.factor_shocks.round(8).tolist(),
        stock_returns=scenario.stock_returns.round(8).tolist(),
        value_path=list(result.value_path.values()) if result else [],
    )


def evaluate_policy(env: StressTestEnv, policy: Policy, seeds: Iterable[int]) -> list[EpisodeRecord]:
    """Run ``policy`` on every seed."""
    return [run_episode(env, policy, s) for s in seeds]


def _stats(x: Sequence[float]) -> dict[str, float | None]:
    if len(x) == 0:
        return {"mean": None, "median": None, "std": None}
    arr = np.asarray(x, dtype=float)
    return {"mean": float(arr.mean()), "median": float(np.median(arr)),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}


def summarize(records: Sequence[EpisodeRecord]) -> dict[str, Any]:
    """Rates, severities, plausibility, reward components and diagnostics for one record set."""
    n = len(records)
    if n == 0:
        raise ValueError("no episodes to summarise")
    def rate(pred) -> float:
        return sum(1 for r in records if pred(r)) / n
    breaches = [r for r in records if r.breached]
    valid = [r for r in records if r.valid_breach]
    pct = np.array([r.historical_percentile for r in records])
    return {
        "n_episodes": n,
        "breach_rate": rate(lambda r: r.breached),
        "valid_breach_rate": rate(lambda r: r.valid_breach),
        "implausible_breach_rate": rate(lambda r: r.implausible_breach),
        "drawdown_breach_rate": rate(lambda r: r.drawdown_breach),
        "loss_breach_rate": rate(lambda r: r.loss_breach),
        "loss_breach_without_drawdown_breach": sum(1 for r in records if r.loss_breach and not r.drawdown_breach),
        "invalid_rate": rate(lambda r: r.termination == "invalid"),
        "horizon_rate": rate(lambda r: r.termination == "horizon"),
        "valid_share_of_breaches": len(valid) / len(breaches) if breaches else None,
        "severity_breaches": _stats([r.path_drawdown for r in breaches]),
        "severity_valid_breaches": _stats([r.path_drawdown for r in valid]),
        "cumulative_loss_all": _stats([r.cumulative_loss for r in records]),
        "mean_abs_action": float(np.mean([r.mean_abs_action for r in records])),
        "episode_return": _stats([r.total_reward for r in records]),
        "episode_length": _stats([r.length for r in records]),
        "reward_components_mean": {k: float(np.mean([r.components.get(k, 0.0) for r in records])) for k in COMPONENTS},
        "historical_percentile_quantiles": {str(q): float(np.quantile(pct, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "breach_day_distribution": {str(d): sum(1 for r in breaches if r.breach_day == d) for d in range(1, 6)},
    }


def diversity(records: Sequence[EpisodeRecord], factor_sd: np.ndarray, factor_names: Sequence[str]) -> dict[str, Any]:
    """Scenario diversity of valid breaches (collapse diagnostics).

    Uses each path's cumulative factor return in model standard deviations.
    """
    valid = [r for r in records if r.valid_breach]
    if len(valid) < 2:
        return {"n_valid": len(valid), "note": "fewer than two valid breaches"}
    cum = np.array([np.prod(1 + np.asarray(r.factor_shocks), axis=0) - 1 for r in valid]) / factor_sd
    diffs = cum[:, None, :] - cum[None, :, :]
    dist = np.sqrt((diffs ** 2).sum(-1))[np.triu_indices(len(valid), 1)]
    rounded = {tuple(np.round(np.asarray(r.actions), 2).ravel()) for r in valid}
    within = [float(np.asarray(r.actions).std(axis=0).mean()) if len(r.actions) > 1 else 0.0 for r in valid]
    most_negative = pd.Series(np.argmin(cum, axis=1)).map(dict(enumerate(factor_names))).value_counts(normalize=True)
    return {
        "n_valid": len(valid),
        "unique_action_sequences_share": len(rounded) / len(valid),
        "mean_pairwise_distance_sd_units": float(dist.mean()),
        "cumulative_factor_shock_sd_units": {
            name: {"mean": float(cum[:, j].mean()), "std": float(cum[:, j].std(ddof=1))}
            for j, name in enumerate(factor_names)},
        "all_factors_negative_share": float((cum < 0).all(axis=1).mean()),
        "most_negative_factor_share": most_negative.round(4).to_dict(),
        "within_path_action_std_mean": float(np.mean(within)),
        "distinct_anchor_dates": len({r.anchor_date for r in valid}),
    }


def hacking_diagnostics(records: Sequence[EpisodeRecord]) -> dict[str, Any]:
    """Signals of reward hacking: saturation, instant extreme breaches, redraws, early termination."""
    breaches = [r for r in records if r.breached]
    by_term: dict[str, list[float]] = {}
    for r in records:
        by_term.setdefault(r.termination, []).append(r.total_reward)
    return {
        "saturated_action_share": float(np.mean([r.saturated_action_share for r in records])),
        "day1_breach_share_of_breaches": (sum(1 for r in breaches if r.breach_day == 1) / len(breaches)) if breaches else None,
        "mean_residual_draws_per_step": float(np.mean([r.residual_draws for r in records])),
        "max_residual_draws_per_step": float(np.max([r.residual_draws for r in records])),
        "mean_return_by_termination": {k: float(np.mean(v)) for k, v in by_term.items()},
        "count_by_termination": {k: len(v) for k, v in by_term.items()},
        "breach_reward_paid_without_valid_breach": sum(
            1 for r in records if r.components.get("breach", 0) > 0 and not r.valid_breach),
        "positive_breach_reward_on_invalid": sum(
            1 for r in records if r.termination == "invalid" and r.components.get("breach", 0) > 0),
    }


KEY_METRICS = ("breach_rate", "valid_breach_rate", "implausible_breach_rate", "invalid_rate",
               "mean_abs_action")


def across_sets(summaries: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean and std (ddof=1) of key metrics across evaluation sets or training seeds."""
    out = {}
    for key in KEY_METRICS:
        vals = [s[key] for s in summaries]
        out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "n": len(vals)}
    rets = [s["episode_return"]["mean"] for s in summaries]
    out["episode_return_mean"] = {"mean": float(np.mean(rets)),
                                  "std": float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0, "n": len(rets)}
    return out
