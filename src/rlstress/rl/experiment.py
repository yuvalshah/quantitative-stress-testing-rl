"""Stage 5 experiment: init -> train (thresholds x seeds) -> validate -> (explicit) final test.

Run directory layout (``results/stage5/run_<UTC timestamp>/``)::

    config.toml                  snapshot used by every step
    run_meta.json                seeds, thresholds, data source, shock-model fit window
    shock_model.json             fitted once on the training split; reused by all steps
    threshold_calibration.json   real-week rejection shares (train, validation)
    training/thr_<q>/seed_<s>/   progress.csv, training_episodes.csv, selection_log.json,
                                 best_model.zip (+ vecnormalize), checkpoints/, final_model.zip
    validation/                  metrics.json, episodes.csv, threshold_comparison.csv,
                                 reward_components.csv, representative_scenarios.json
    selection.json               selected threshold and checkpoint (validation only)
    summary.json, stage5_report.md
    test/                        written only by the explicit final-test step (once)
"""

from __future__ import annotations

import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rlstress.config import ProjectConfig, load_config
from rlstress.data.sources import utc_now_iso
from rlstress.data.storage import read_frame, write_json
from rlstress.portfolio import Portfolio, calculate_asset_returns
from rlstress.rl.calibration import real_window_percentiles, select_threshold, threshold_calibration
from rlstress.rl.evaluation import (
    COMPONENTS,
    across_sets,
    diversity,
    evaluate_policy,
    hacking_diagnostics,
    summarize,
)
from rlstress.rl.policies import RandomPolicy, scripted_decline_policy, zero_policy
from rlstress.rl.training import build_env, load_policy, train_ppo
from rlstress.shocks import ShockModel, fit_shock_model
from rlstress.stress import evaluate_return_path, replay_historical_scenario

logger = logging.getLogger(__name__)


def _thr_dir(q: float) -> str:
    return f"thr_{q:g}"


def apply_overrides(config: ProjectConfig, overrides: dict[str, Any]) -> ProjectConfig:
    """Return ``config`` with ``{"section.field": value}`` overrides applied and re-validated.

    Lists become tuples. Used to record controlled follow-up experiments in the run metadata.
    """
    import dataclasses

    from rlstress.config import validate_config

    for key, value in overrides.items():
        section, field = key.split(".", 1)
        part = getattr(config, section)
        if field not in {f.name for f in dataclasses.fields(part)}:
            raise ValueError(f"unknown config field {key}")
        value = tuple(value) if isinstance(value, list) else value
        config = dataclasses.replace(config, **{section: dataclasses.replace(part, **{field: value})})
    validate_config(config)
    return config


def init_run(config_path: Path, returns_path: Path, out_root: Path, label: str = "",
             overrides: dict[str, Any] | None = None) -> Path:
    """Create a unique run directory with a config snapshot, the train-fitted shock model and calibration."""
    overrides = overrides or {}
    config = apply_overrides(load_config(config_path), overrides)
    source = read_frame(returns_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_root / (f"run_{stamp}" + (f"_{label}" if label else ""))
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy(config_path, run_dir / "config.toml")
    start, end = config.data.splits.as_dict()[config.simulation.fit_window]
    model = fit_shock_model(source, config.portfolio.tickers, config.simulation, start, end,
                            config.portfolio.return_column_template)
    if model.fit_end > config.data.splits.train[1]:
        raise RuntimeError("shock model fitted beyond the training split")
    write_json(model.to_dict(), run_dir / "shock_model.json")
    splits = config.data.splits.as_dict()
    calib = threshold_calibration(model, source, {"train": splits["train"], "validation": splits["validation"]},
                                  config.training.plausibility_thresholds, config.simulation.horizon_days)
    write_json(calib, run_dir / "threshold_calibration.json")
    write_json({"created_utc": utc_now_iso(), "returns_path": str(returns_path), "config_path": str(config_path),
                "overrides": overrides,
                "seeds": list(config.training.seeds), "thresholds": list(config.training.plausibility_thresholds),
                "total_timesteps": config.training.total_timesteps,
                "shock_model_fit": [model.fit_start, model.fit_end], "splits": {k: list(v) for k, v in splits.items()}},
               run_dir / "run_meta.json")
    return run_dir


def load_run(run_dir: Path) -> tuple[ProjectConfig, pd.DataFrame, ShockModel, dict[str, Any]]:
    """Config snapshot, data, shock model and metadata of a run."""
    meta = json.loads((run_dir / "run_meta.json").read_text())
    config = apply_overrides(load_config(run_dir / "config.toml"), meta.get("overrides", {}))
    source = read_frame(Path(meta["returns_path"]))
    model = ShockModel.from_dict(json.loads((run_dir / "shock_model.json").read_text()))
    return config, source, model, meta


def train_one(run_dir: Path, threshold: float, seed: int, total_timesteps: int | None = None) -> dict[str, Any]:
    """Train one (threshold, seed) in its own directory (safe to call from a worker process)."""
    logging.basicConfig(level=logging.INFO)
    config, source, model, _ = load_run(run_dir)
    out = run_dir / "training" / _thr_dir(threshold) / f"seed_{seed}"
    if (out / "train_meta.json").exists():
        raise FileExistsError(f"{out} already trained")
    res = train_ppo(config, source, model, threshold, seed, out, total_timesteps)
    return {"threshold": threshold, "seed": seed, "seconds": res.seconds, "best": res.best}


def train_all(run_dir: Path, jobs: int = 1, total_timesteps: int | None = None) -> list[dict[str, Any]]:
    """Train every configured (threshold, seed); ``jobs > 1`` uses separate processes."""
    config, _, _, _ = load_run(run_dir)
    tasks = [(q, s) for q in config.training.plausibility_thresholds for s in config.training.seeds]
    if jobs <= 1:
        return [train_one(run_dir, q, s, total_timesteps) for q, s in tasks]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(train_one, run_dir, q, s, total_timesteps) for q, s in tasks]
        return [f.result() for f in futures]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _policies(config: ProjectConfig, run_dir: Path, threshold: float, env) -> list:
    k = env.simulator.model.k
    t = config.training
    pols = [zero_policy(k), scripted_decline_policy(k, t.scripted_action),
            RandomPolicy(k, t.random_baseline_seed_offset)]
    for seed in t.seeds:
        d = run_dir / "training" / _thr_dir(threshold) / f"seed_{seed}"
        if (d / "best_model.zip").exists():
            pols.append(load_policy(d, env, "best", name=f"ppo_seed_{seed}"))
    return pols


def learning_curves(run_dir: Path, threshold: float, seeds: list[int]) -> dict[str, Any]:
    """Final SB3 training diagnostics and training-episode trends per seed."""
    out = {}
    for seed in seeds:
        d = run_dir / "training" / _thr_dir(threshold) / f"seed_{seed}"
        if not (d / "progress.csv").exists():
            continue
        prog = pd.read_csv(d / "progress.csv")
        last = prog.iloc[-1]
        eps = pd.read_csv(d / "training_episodes.csv")
        q = max(1, len(eps) // 10)
        def share(df: pd.DataFrame, term: str) -> float:
            return float((df["termination"] == term).mean()) if len(df) else float("nan")
        out[str(seed)] = {
            "final": {k: (float(last[k]) if k in last and pd.notna(last[k]) else None) for k in
                      ("train/entropy_loss", "train/approx_kl", "train/explained_variance", "train/value_loss",
                       "train/clip_fraction", "train/std")},
            "first_decile": {"mean_return": float(eps["total_reward"][:q].mean()),
                             "valid_breach_share": share(eps[:q], "valid_breach")},
            "last_decile": {"mean_return": float(eps["total_reward"][-q:].mean()),
                            "valid_breach_share": share(eps[-q:], "valid_breach")},
            "n_training_episodes": int(len(eps)),
            "selection_log": json.loads((d / "selection_log.json").read_text()),
            "train_meta": json.loads((d / "train_meta.json").read_text()),
        }
    return out


def historical_analogue(model: ShockModel, source: pd.DataFrame, portfolio: Portfolio, config: ProjectConfig,
                        factor_path: np.ndarray) -> dict[str, Any]:
    """Closest actual train/validation window (same length) by cumulative factor shock in model sd units."""
    length = len(factor_path)
    sd = model.factor_sd
    target = (np.prod(1 + factor_path, axis=0) - 1) / sd
    end = config.data.splits.validation[1]
    f = source.loc[:end, list(model.factors)].dropna()
    vals = f.to_numpy()
    best_i, best_d = None, np.inf
    for i in range(len(vals) - length + 1):
        cum = (np.prod(1 + vals[i:i + length], axis=0) - 1) / sd
        d = float(np.sqrt(((cum - target) ** 2).sum()))
        if d < best_d:
            best_i, best_d = i, d
    start = f.index[best_i]
    asset = calculate_asset_returns(source, portfolio.tickers, config.portfolio.return_column_template)
    replay = replay_historical_scenario(portfolio, asset, start.date().isoformat(), length)
    window = f.iloc[best_i:best_i + length]
    pct = real_window_percentiles(model, window, window.index[0].date().isoformat(),
                                  window.index[-1].date().isoformat(), length)
    return {"start_date": start.date().isoformat(), "end_date": window.index[-1].date().isoformat(),
            "distance_sd_units": best_d,
            "cumulative_factor_returns": dict(zip(model.factors, (np.prod(1 + window.to_numpy(), axis=0) - 1).tolist())),
            "portfolio_return": replay.portfolio_return, "path_max_drawdown": replay.max_drawdown,
            "historical_percentile": float(pct.iloc[0]) if len(pct) else None,
            "split": "train" if start <= pd.Timestamp(config.data.splits.train[1]) else "validation"}


def representative_scenarios(records: list, model: ShockModel, source: pd.DataFrame, config: ProjectConfig,
                             n: int) -> list[dict[str, Any]]:
    """Most severe, median-severity and most plausible valid breaches with Stage 3 detail and analogues."""
    valid = [r for r in records if r.valid_breach]
    if not valid:
        return []
    by_sev = sorted(valid, key=lambda r: r.path_drawdown)
    picks = {"most_severe": by_sev[-1], "median_severity": by_sev[len(by_sev) // 2],
             "most_plausible": max(valid, key=lambda r: r.historical_percentile)}
    portfolio = Portfolio.from_config(config.portfolio)
    out = []
    for label, r in list(picks.items())[:n]:
        stock = np.asarray(r.stock_returns)
        dates = pd.bdate_range(pd.Timestamp(r.anchor_date) + pd.offsets.BDay(1), periods=len(stock))
        res = evaluate_return_path(portfolio, pd.DataFrame(stock, index=dates, columns=list(portfolio.tickers)),
                                   pd.Timestamp(r.anchor_date), config.risk.limits, label, "simulated_factor_shock")
        out.append({
            "label": label, "policy": r.policy, "episode_seed": r.episode_seed, "anchor_date": r.anchor_date,
            "breach_day": r.breach_day, "five_day_or_breach_loss": r.cumulative_loss,
            "path_max_drawdown": r.path_drawdown, "historical_percentile": r.historical_percentile,
            "path_log_tail_probability": r.path_log_tail_probability,
            "actions": r.actions,
            "daily_factor_shocks": [dict(zip(model.factors, d)) for d in r.factor_shocks],
            "daily_stock_returns": [dict(zip(portfolio.tickers, d)) for d in r.stock_returns],
            "value_path": r.value_path,
            "asset_contributions": res.asset_contributions,
            "limit_results_stage3_strict_horizon": [x.to_dict() for x in res.limit_results],
            "historical_analogue": historical_analogue(model, source, portfolio, config, np.asarray(r.factor_shocks)),
            "note": "dates after the anchor are step labels for a simulated path, not historical dates",
        })
    return out


def validate_run(run_dir: Path) -> dict[str, Any]:
    """Validation-period evaluation of all policies at all thresholds, selection, report."""
    config, source, model, meta = load_run(run_dir)
    t = config.training
    out_dir = run_dir / "validation"
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; validation results are not overwritten")
    out_dir.mkdir()
    calib = json.loads((run_dir / "threshold_calibration.json").read_text())
    rows, metrics = [], {}
    for q in t.plausibility_thresholds:
        env = build_env(config, source, model, q, "validation")
        metrics[str(q)] = {}
        for pol in _policies(config, run_dir, q, env):
            per_set, all_records = [], []
            for base in t.eval_seed_sets:
                recs = evaluate_policy(env, pol, range(base, base + t.validation_episodes))
                per_set.append(summarize(recs))
                all_records += recs
            for r in all_records:
                rows.append({"threshold": q, **r.flat()})
            metrics[str(q)][pol.name] = {
                "per_set": per_set, "across_sets": across_sets(per_set), "pooled": summarize(all_records),
                "diversity": diversity(all_records, model.factor_sd, model.factors),
                "hacking": hacking_diagnostics(all_records),
            }
            if pol.name.startswith("ppo"):
                metrics[str(q)][pol.name]["_records"] = all_records
        ppo_names = [n for n in metrics[str(q)] if n.startswith("ppo")]
        if ppo_names:
            seed_summ = [metrics[str(q)][n]["pooled"] for n in ppo_names]
            metrics[str(q)]["ppo_across_training_seeds"] = across_sets(seed_summ)
        metrics[str(q)]["learning"] = learning_curves(run_dir, q, list(t.seeds))

    choice = select_threshold(calib, t.plausibility_thresholds, config.simulation.min_plausibility)
    q_sel = choice["selected"]
    seed_best = {}
    for seed in t.seeds:
        lc = metrics[str(q_sel)]["learning"].get(str(seed))
        if lc:
            b = lc["train_meta"]["best"]
            seed_best[seed] = (b["valid_breach_rate"], b["mean_return"], b["timesteps"])
    selected_seed = max(seed_best, key=lambda s: seed_best[s][:2]) if seed_best else None
    selection = {"threshold": choice, "selected_seed": selected_seed,
                 "selected_checkpoint": str(run_dir / "training" / _thr_dir(q_sel) / f"seed_{selected_seed}" / "best_model.zip")
                 if selected_seed is not None else None,
                 "checkpoint_rule": "per seed: best selection-set (validation anchors, seeds disjoint from reporting) "
                                    "valid-breach rate, tie -> mean return; across seeds: same rule on the selection set",
                 "seed_selection_scores": {str(k): {"valid_breach_rate": v[0], "mean_return": v[1], "timesteps": v[2]}
                                           for k, v in seed_best.items()},
                 "uses_test_data": False}
    reps = []
    if selected_seed is not None:
        recs = metrics[str(q_sel)][f"ppo_seed_{selected_seed}"]["_records"]
        reps = representative_scenarios(recs, model, source, config, t.n_representative_scenarios)
    for q in metrics:
        for v in metrics[q].values():
            if isinstance(v, dict):
                v.pop("_records", None)

    episodes = pd.DataFrame(rows)
    episodes.to_csv(out_dir / "episodes.csv", index=False)
    comp_cols = [c for c in episodes.columns if c.startswith("reward_")]
    episodes.groupby(["threshold", "policy"])[comp_cols + ["total_reward"]].mean().to_csv(out_dir / "reward_components.csv")
    comparison = []
    for q in t.plausibility_thresholds:
        for name, m in metrics[str(q)].items():
            if name in ("learning", "ppo_across_training_seeds"):
                continue
            a = m["across_sets"]
            comparison.append({"threshold": q, "policy": name,
                               "validation_real_week_rejected_share": calib["validation"]["rejected_share"][str(q)],
                               **{f"{k}_mean": v["mean"] for k, v in a.items()},
                               **{f"{k}_std": v["std"] for k, v in a.items()},
                               "n_episodes": m["pooled"]["n_episodes"]})
    pd.DataFrame(comparison).to_csv(out_dir / "threshold_comparison.csv", index=False)
    write_json(metrics, out_dir / "metrics.json")
    write_json(reps, out_dir / "representative_scenarios.json")
    write_json(selection, run_dir / "selection.json")
    summary = {"generated_utc": utc_now_iso(), "run_dir": str(run_dir), "meta": meta, "calibration": calib,
               "selection": selection, "comparison": comparison}
    write_json(summary, run_dir / "summary.json")
    (run_dir / "stage5_report.md").write_text(render_report(config, meta, calib, metrics, selection, reps))
    return summary


# ---------------------------------------------------------------------------
# Final test (explicit, once)
# ---------------------------------------------------------------------------


def final_test(run_dir: Path, confirm: bool = False, force: bool = False) -> dict[str, Any]:
    """Evaluate the selected configuration on test-period anchors. Requires ``confirm=True``; runs once."""
    if not confirm:
        raise PermissionError("final test requires explicit confirmation (--confirm-final-test)")
    sel_path = run_dir / "selection.json"
    if not sel_path.exists():
        raise FileNotFoundError("run validation and selection first")
    out_dir = run_dir / "test"
    if (out_dir / "test_evaluation.json").exists() and not force:
        raise FileExistsError("final test already run for this selection; repeated test runs are not allowed")
    config, source, model, _ = load_run(run_dir)
    selection = json.loads(sel_path.read_text())
    q = selection["threshold"]["selected"]
    seed = selection["selected_seed"]
    t = config.training
    env = build_env(config, source, model, q, "test")
    pols = _policies(config, run_dir, q, env)
    pols = [p for p in pols if not p.name.startswith("ppo") or p.name == f"ppo_seed_{seed}"]
    seeds = range(t.test_seed, t.test_seed + t.test_episodes)
    results = {}
    for pol in pols:
        recs = evaluate_policy(env, pol, seeds)
        results[pol.name] = {"summary": summarize(recs), "diversity": diversity(recs, model.factor_sd, model.factors)}
    splits = config.data.splits.as_dict()
    calib = threshold_calibration(model, source, {"test": splits["test"]}, [q], config.simulation.horizon_days)
    out_dir.mkdir(exist_ok=True)
    payload = {"generated_utc": utc_now_iso(), "selection": selection, "threshold": q,
               "n_episodes": t.test_episodes, "results": results, "test_real_week_calibration": calib}
    write_json(payload, out_dir / "test_evaluation.json")
    return payload


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _p(x: float | None, d: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def _ms(a: dict[str, float], pct: bool = True) -> str:
    return f"{_p(a['mean'])} ± {_p(a['std'])}" if pct else f"{a['mean']:.2f} ± {a['std']:.2f}"


def render_report(config: ProjectConfig, meta: dict[str, Any], calib: dict[str, Any], metrics: dict[str, Any],
                  selection: dict[str, Any], reps: list[dict[str, Any]]) -> str:
    """Markdown report of the validation experiment (no test results)."""
    t = config.training
    lines = ["# Stage 5 PPO Validation Report", "",
             f"Run created {meta['created_utc']}. Config overrides: {meta.get('overrides') or 'none'}. Shock model fitted {meta['shock_model_fit'][0]} to "
             f"{meta['shock_model_fit'][1]} (training split). PPO: {t.total_timesteps:,} timesteps per run, "
             f"seeds {list(t.seeds)}, thresholds {list(t.plausibility_thresholds)}. "
             f"Validation anchors {meta['splits']['validation'][0]} to {meta['splits']['validation'][1]}; "
             f"{len(t.eval_seed_sets)} evaluation sets x {t.validation_episodes} episodes per policy "
             f"(identical episode seeds for every policy). **No test-period data used.**", "",
             "## Threshold calibration on real weeks", "",
             "Threshold q rejects a path whose historical percentile (share of real training 5-day windows at "
             "least as extreme) is below q.", "",
             "| q | train weeks rejected | validation weeks rejected | validation / nominal |", "|---|---|---|---|"]
    for q in t.plausibility_thresholds:
        lines.append(f"| {q:g} | {_p(calib['train']['rejected_share'][str(q)], 2)} | "
                     f"{_p(calib['validation']['rejected_share'][str(q)], 2)} | "
                     f"{calib['validation_calibration_ratio'][str(q)]:.2f} |")
    lines += ["", f"Selected threshold: **{selection['threshold']['selected']:g}** "
              f"({selection['threshold']['rule']}).", ""]
    for q in t.plausibility_thresholds:
        m = metrics[str(q)]
        lines += [f"## Validation results, q = {q:g}", "",
                  "| Policy | breach | valid breach | implausible | invalid | mean |a| | return |",
                  "|---|---|---|---|---|---|---|"]
        for name, v in m.items():
            if name in ("learning", "ppo_across_training_seeds"):
                continue
            a = v["across_sets"]
            lines.append(f"| {name} | {_ms(a['breach_rate'])} | {_ms(a['valid_breach_rate'])} | "
                         f"{_ms(a['implausible_breach_rate'])} | {_ms(a['invalid_rate'])} | "
                         f"{a['mean_abs_action']['mean']:.3f} | {_ms(a['episode_return_mean'], False)} |")
        if "ppo_across_training_seeds" in m:
            a = m["ppo_across_training_seeds"]
            lines.append(f"| PPO across training seeds | {_ms(a['breach_rate'])} | {_ms(a['valid_breach_rate'])} | "
                         f"{_ms(a['implausible_breach_rate'])} | {_ms(a['invalid_rate'])} | "
                         f"{a['mean_abs_action']['mean']:.3f} | {_ms(a['episode_return_mean'], False)} |")
        lines += ["", "± = std across the 3 evaluation sets (per policy) or across training seeds (last row).", ""]
    q = selection["threshold"]["selected"]
    m = metrics[str(q)]
    lines += [f"## Diagnostics at the selected threshold (q = {q:g})", "",
              "| Policy | valid-breach severity (mean DD) | day-1 breaches | saturated actions | unique action seqs "
              "| all-factors-negative | mean pairwise dist (sd) | breach reward on non-valid |",
              "|---|---|---|---|---|---|---|---|"]
    for name, v in m.items():
        if name in ("learning", "ppo_across_training_seeds"):
            continue
        d, h, p = v["diversity"], v["hacking"], v["pooled"]
        sev = p["severity_valid_breaches"]["mean"]
        lines.append(f"| {name} | {_p(sev)} | {h['day1_breach_share_of_breaches'] if h['day1_breach_share_of_breaches'] is None else _p(h['day1_breach_share_of_breaches'])} | "
                     f"{_p(h['saturated_action_share'])} | {_p(d.get('unique_action_sequences_share'))} | "
                     f"{_p(d.get('all_factors_negative_share'))} | "
                     f"{d.get('mean_pairwise_distance_sd_units', float('nan')):.2f} | "
                     f"{h['breach_reward_paid_without_valid_breach']} |")
    lines += ["", "Mean reward components per episode:", "",
              "| Policy | " + " | ".join(COMPONENTS) + " |",
              "|---" * (len(COMPONENTS) + 1) + "|"]
    for name, v in m.items():
        if name in ("learning", "ppo_across_training_seeds"):
            continue
        c = v["pooled"]["reward_components_mean"]
        lines.append(f"| {name} | " + " | ".join(f"{c[k]:.3f}" for k in
                                                  COMPONENTS) + " |")
    lines += ["", "## Learning (selected threshold)", "",
              "| Seed | return first→last decile | valid share first→last | best selection valid rate @ step "
              "| entropy | approx KL | explained var |", "|---|---|---|---|---|---|---|"]
    for seed, lc in m["learning"].items():
        b = lc["train_meta"]["best"]
        f = lc["final"]
        def fmt(x): return "n/a" if x is None else f"{x:.3g}"
        lines.append(f"| {seed} | {lc['first_decile']['mean_return']:.2f} → {lc['last_decile']['mean_return']:.2f} | "
                     f"{_p(lc['first_decile']['valid_breach_share'])} → {_p(lc['last_decile']['valid_breach_share'])} | "
                     f"{_p(b['valid_breach_rate'])} @ {b['timesteps']:,} | {fmt(f['train/entropy_loss'])} | "
                     f"{fmt(f['train/approx_kl'])} | {fmt(f['train/explained_variance'])} |")
    lines += ["", f"Selected checkpoint: seed {selection['selected_seed']} "
              f"(`{selection['selected_checkpoint']}`).", "", "## Representative discovered scenarios", ""]
    for r in reps:
        ha = r["historical_analogue"]
        shocks = "; ".join(f"d{i + 1}: " + ", ".join(f"{k.split('_')[0]} {v:+.2%}" for k, v in d.items())
                           for i, d in enumerate(r["daily_factor_shocks"]))
        lines += [f"**{r['label']}** (anchor {r['anchor_date']}, breach day {r['breach_day']}): "
                  f"loss {_p(r['five_day_or_breach_loss'], 2)}, path DD {_p(r['path_max_drawdown'], 2)}, "
                  f"historical percentile {r['historical_percentile']:.3f}.  ",
                  f"Factor shocks: {shocks}.  ",
                  f"Closest real window: {ha['start_date']} to {ha['end_date']} ({ha['split']}), "
                  f"portfolio {_p(ha['portfolio_return'], 2)}, DD {_p(ha['path_max_drawdown'], 2)}, "
                  f"percentile {ha['historical_percentile']:.3f}, distance {ha['distance_sd_units']:.2f} sd.", ""]
    if not reps:
        lines.append("_No valid breaches by the selected policy; nothing to show._")
    return "\n".join(lines) + "\n"
