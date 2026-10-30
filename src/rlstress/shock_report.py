"""Stage 4 orchestration: fit the shock model, validate it and run scripted episodes.

Writes ``shock_model.json`` (parameters), ``shock_model_diagnostics.json``,
``validation_episodes.json`` and ``stage4_report.md``. No network access and no
RL training.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rlstress.config import ProjectConfig
from rlstress.data.sources import utc_now_iso
from rlstress.data.storage import write_json
from rlstress.env import StressTestEnv, make_env
from rlstress.portfolio import Portfolio
from rlstress.shocks import ShockModel, ShockSimulator, fit_shock_model, training_matrix
from rlstress.shocks.diagnostics import (
    candidate_factor_diagnostics,
    mahalanobis_calibration,
    simulated_vs_empirical,
)

OUTPUT_FILES = ("shock_model.json", "shock_model_diagnostics.json", "validation_episodes.json",
                "stage4_report.md")


def scripted_episodes(config: ProjectConfig, source: pd.DataFrame, model: ShockModel,
                      anchor: str | None = None) -> list[dict[str, Any]]:
    """Deterministic validation episodes (residual mode ``mean``) for fixed factor actions."""
    cfg = dataclasses.replace(config, simulation=dataclasses.replace(config.simulation, residual_mode="mean"))
    env = make_env(cfg, source, model)
    k = model.k
    names = list(model.factors)
    def one_hot(name: str, v: float) -> np.ndarray:
        a = np.zeros(k)
        a[names.index(name)] = v
        return a
    scripts = {
        "zero_shock": np.zeros(k),
        "broad_crash_moderate (-0.15 all factors)": np.full(k, -0.15),
        "broad_crash_severe (-0.5 all factors)": np.full(k, -0.5),
        "market_only (-0.3 SPY)": one_hot(model.market_factor, -0.3),
        "financials_only (-0.3 XLF)": one_hot("XLF_return", -0.3) if "XLF_return" in names else np.zeros(k),
        "incoherent (SPY -0.2, QQQ +0.2)": (one_hot(model.market_factor, -0.2)
                                            + (one_hot("QQQ_return", 0.2) if "QQQ_return" in names else 0)),
    }
    anchor = anchor or env.market_states.index[len(env.market_states) // 2].date().isoformat()
    out = []
    for name, action in scripts.items():
        env.reset(seed=config.simulation.seed, options={"anchor_date": anchor})
        total, comps, info = 0.0, {}, {}
        for _ in range(env.horizon):
            _, r, term, _, info = env.step(action.astype(np.float32))
            total += r
            for key, v in info["reward_components"].items():
                comps[key] = comps.get(key, 0.0) + v
            if term:
                break
        out.append({
            "episode": name, "anchor_date": anchor, "action": action.tolist(),
            "daily_factor_shock": dict(zip(names, (action * env.simulator.bounds).tolist())),
            "steps": info.get("step"), "cumulative_loss": info.get("cumulative_loss"),
            "path_drawdown": info.get("path_drawdown"), "breached": info.get("breached"),
            "valid_breach": info.get("valid_breach"),
            "path_historical_percentile": info.get("path_historical_percentile"),
            "path_log_tail_probability": info.get("path_log_tail_probability"),
            "total_reward": total, "reward_components": comps,
        })
    return out


def random_policy_baseline(env: StressTestEnv, episodes: int, seed: int) -> dict[str, Any]:
    """Uniform random actions: breach and valid-breach rates (a lower bound for any learned policy)."""
    env.action_space.seed(seed)
    breaches = valid = invalid = 0
    rewards = []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        done, total, info = False, 0.0, {}
        while not done:
            _, r, done, _, info = env.step(env.action_space.sample())
            total += r
        breaches += bool(info.get("breached"))
        valid += bool(info.get("valid_breach"))
        invalid += bool(info.get("invalid"))
        rewards.append(total)
    return {"episodes": episodes, "seed": seed, "breach_rate": breaches / episodes,
            "valid_breach_rate": valid / episodes, "invalid_rate": invalid / episodes,
            "mean_reward": float(np.mean(rewards))}


def run_stage4(config: ProjectConfig, source: pd.DataFrame, output_dir: Path, source_label: str,
               overwrite: bool = False, baseline_episodes: int = 200) -> dict[str, Any]:
    """Fit, validate and document the shock model and environment."""
    output_dir = Path(output_dir)
    existing = [f for f in OUTPUT_FILES if (output_dir / f).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{output_dir} already contains {existing}; pass overwrite=True (--overwrite)")
    sim = config.simulation
    start, end = config.data.splits.as_dict()[sim.fit_window]
    portfolio = Portfolio.from_config(config.portfolio)
    model = fit_shock_model(source, portfolio.tickers, sim, start, end, config.portfolio.return_column_template)
    simulator = ShockSimulator(model, sim)
    factors, stocks, _ = training_matrix(source, portfolio.tickers, sim.factors, start, end,
                                         config.portfolio.return_column_template)
    diagnostics = {
        "generated_at_utc": utc_now_iso(),
        "data_source": source_label,
        "fit_window": {"name": sim.fit_window, "configured": [start, end],
                       "actual": [model.fit_start, model.fit_end], "n_obs": model.n_obs,
                       "n_excluded_incomplete": model.n_excluded},
        "model": {"nu_joint": model.nu, "nu_plausibility": model.plaus_nu, "shrinkage_joint": model.shrinkage,
                  "shrinkage_plausibility": model.plaus_shrinkage, "loglik": model.loglik},
        "factor_action_bounds": dict(zip(model.factors, simulator.bounds.tolist())),
        "factor_model_sd": dict(zip(model.factors, model.factor_sd.tolist())),
        "loadings_on_orthogonalised_factors": pd.DataFrame(model.beta, index=model.tickers,
                                                           columns=model.factors).round(4).to_dict(),
        "loadings_on_raw_factors": pd.DataFrame(model.raw_loadings, index=model.tickers,
                                                columns=model.factors).round(4).to_dict(),
        "fit_diagnostics": model.diagnostics,
        "calibration_plausibility_model": mahalanobis_calibration(model, factors, "plausibility"),
        "calibration_joint_model": mahalanobis_calibration(model, factors, "joint"),
        "simulated_vs_empirical": simulated_vs_empirical(
            simulator, stocks, portfolio.weights, portfolio.initial_capital,
            sim.diagnostic_simulations, sim.seed, config.risk.confidence_levels),
        "candidate_factors": candidate_factor_diagnostics(model, source, sim.candidate_factors, start, end,
                                                          config.portfolio.return_column_template),
    }
    env = make_env(config, source, model)
    episodes = scripted_episodes(config, source, model)
    baseline = random_policy_baseline(env, baseline_episodes, sim.seed)
    payload = {"generated_at_utc": diagnostics["generated_at_utc"], "environment": {
        "observation_names": env.observation_names, "action_factors": list(model.factors),
        "horizon": env.horizon, "path_limits": [l.name for l in env.path_limits],
        "anchors": {"window": sim.anchor_window, "n_dates": len(env.market_states)},
        "reward_weights": dataclasses.asdict(config.environment),
        "min_plausibility": sim.min_plausibility}, "scripted_episodes": episodes,
        "random_policy_baseline": baseline}

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(model.to_dict(), output_dir / "shock_model.json")
    write_json(diagnostics, output_dir / "shock_model_diagnostics.json")
    write_json(payload, output_dir / "validation_episodes.json")
    (output_dir / "stage4_report.md").write_text(render_report(diagnostics, payload))
    return {"model": model, "diagnostics": diagnostics, "episodes": payload,
            "outputs": {f: str(output_dir / f) for f in OUTPUT_FILES}}


def _pct(x: float | None, d: int = 2) -> str:
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def render_report(diag: dict[str, Any], episodes: dict[str, Any]) -> str:
    """Markdown summary of the Stage 4 fit, validation and scripted episodes."""
    fw, md, fd = diag["fit_window"], diag["model"], diag["fit_diagnostics"]
    lines = [
        "# Stage 4 Shock Model and Environment Report", "",
        f"Generated (UTC): {diag['generated_at_utc']}  ",
        f"Data: `{diag['data_source']}`; fit window **{fw['name']}** {fw['actual'][0]} to {fw['actual'][1]} "
        f"({fw['n_obs']} complete days, {fw['n_excluded_incomplete']} excluded)", "",
        f"Joint Student-t (factors + residuals): nu = {md['nu_joint']}, shrinkage {md['shrinkage_joint']:.4f}. "
        f"Factor plausibility t: nu = {md['nu_plausibility']}, shrinkage {md['shrinkage_plausibility']:.4f}.", "",
        "## Factor model", "",
        "| Stock | R^2 | Residual sd (daily) | Residual excess kurtosis | Lag-1 autocorr |", "|---|---|---|---|---|",
    ]
    for t, r2 in fd["r_squared"].items():
        lines.append(f"| {t} | {r2:.3f} | {_pct(fd['residual_sd_daily'][t], 3)} | "
                     f"{fd['residual_excess_kurtosis'][t]:.2f} | {fd['residual_lag1_autocorr'][t]:.3f} |")
    lines += ["", f"Residual correlation: max |rho| {fd['residual_corr_max_abs']:.3f}, mean |rho| "
              f"{fd['residual_corr_mean_abs']:.3f} (kept in the joint scatter).", "",
              "Multicollinearity: VIF raw " + ", ".join(f"{k} {v:.1f}" for k, v in fd["vif_raw_factors"].items())
              + "; orthogonalised " + ", ".join(f"{k} {v:.2f}" for k, v in fd["vif_orthogonalised_factors"].items())
              + f". Condition number {fd['condition_number_raw_factor_cov']:.0f} -> "
              f"{fd['condition_number_orth_factor_cov']:.1f}.", "",
              "Action bounds (|a| = 1): " + ", ".join(f"{k} ±{_pct(v)}" for k, v in diag["factor_action_bounds"].items()),
              "", "## Calibration on training factor days", "",
              "| Model | nu | p<5% | p<1% | p<0.1% |", "|---|---|---|---|---|"]
    for key in ("calibration_plausibility_model", "calibration_joint_model"):
        c = diag[key]
        rates = {r["level"]: r["empirical_share"] for r in c["tail_rates"]}
        lines.append(f"| {c['model']} | {c['nu']} | {_pct(rates[0.05])} | {_pct(rates[0.01])} | {_pct(rates[0.001])} |")
    sv = diag["simulated_vs_empirical"]
    lines += ["", f"## Simulated ({sv['n_simulated_days']} days) vs empirical training returns", "",
              f"Portfolio daily sd: model {_pct(sv['portfolio_sd_model'], 3)}, empirical {_pct(sv['portfolio_sd_empirical'], 3)}", "",
              "| Confidence | VaR model | VaR empirical | CVaR model | CVaR empirical |", "|---|---|---|---|---|"]
    for p in sv["portfolio"]:
        lines.append(f"| {p['confidence']:.0%} | {_pct(p['var_model'])} | {_pct(p['var_empirical'])} | "
                     f"{_pct(p['cvar_model'])} | {_pct(p['cvar_empirical'])} |")
    lines += ["", "| Stock | sd model | sd emp. | 1% q model | 1% q emp. |", "|---|---|---|---|---|"]
    for s in sv["per_stock"]:
        lines.append(f"| {s['ticker']} | {_pct(s['sd_model'], 3)} | {_pct(s['sd_empirical'], 3)} | "
                     f"{_pct(s['q01_model'])} | {_pct(s['q01_empirical'])} |")
    lines += ["", "## Candidate factors (not in the model)", "",
              "| Factor | n | mean incremental R^2 | max incremental R^2 | corr with market |", "|---|---|---|---|---|"]
    for c in diag["candidate_factors"]:
        if "mean_incremental_r2" in c:
            lines.append(f"| {c['factor']} | {c['n_obs']} | {c['mean_incremental_r2']:.4f} | "
                         f"{c['max_incremental_r2']:.4f} | {c['corr_with_market']:.2f} |")
    lines += ["", "## Scripted validation episodes (residual mode: mean)", "",
              "| Episode | Steps | Loss | Path DD | Breach | Real-window percentile | Valid | Reward |",
              "|---|---|---|---|---|---|---|---|"]
    for e in episodes["scripted_episodes"]:
        lines.append(f"| {e['episode']} | {e['steps']} | {_pct(e['cumulative_loss'])} | {_pct(e['path_drawdown'])} | "
                     f"{e['breached']} | {e['path_historical_percentile']:.4f} | {e['valid_breach']} | "
                     f"{e['total_reward']:.2f} |")
    b = episodes["random_policy_baseline"]
    lines += ["", f"Random uniform policy ({b['episodes']} episodes): breach rate {_pct(b['breach_rate'], 1)}, "
              f"valid breach rate {_pct(b['valid_breach_rate'], 1)}, invalid {_pct(b['invalid_rate'], 1)}, "
              f"mean reward {b['mean_reward']:.2f}.", ""]
    return "\n".join(lines)
