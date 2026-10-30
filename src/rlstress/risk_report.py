"""Stage 3 orchestration: run the risk engine on Stage 2 data and write results.

Calculation code lives in :mod:`rlstress.portfolio`, :mod:`rlstress.risk` and
:mod:`rlstress.stress`; this module only selects windows, calls them, and
serialises results with full metadata. It performs no network access.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from rlstress.config import ProjectConfig
from rlstress.data.storage import write_frame, write_json
from rlstress.data.sources import utc_now_iso
from rlstress.portfolio import (
    Portfolio,
    apply_missing_data_policy,
    base_date_for,
    buy_and_hold_value,
    calculate_asset_returns,
    calculate_portfolio_returns,
    calculate_portfolio_value,
    select_window,
    value_consistency_error,
)
from rlstress.risk import (
    MetricObservation,
    calculate_asset_risk_contributions,
    calculate_historical_cvar,
    calculate_historical_var,
    calculate_historical_volatility,
    calculate_max_drawdown,
    drawdown_series,
    evaluate_risk_limits,
    return_contributions,
    rolling_volatility_table,
)
from rlstress.risk.cvar import CVAR_ESTIMATOR
from rlstress.risk.var import VAR_ESTIMATOR
from rlstress.stress import (
    StressResult,
    apply_asset_return_shocks,
    replay_historical_scenario,
    worst_historical_windows,
)

logger = logging.getLogger(__name__)

OUTPUT_FILES = (
    "portfolio_returns.csv",
    "portfolio_value.csv",
    "historical_risk_metrics.json",
    "asset_risk_contributions.csv",
    "asset_return_correlation.csv",
    "extreme_loss_contributions.csv",
    "stress_scenario_results.json",
    "risk_engine_report.md",
)

CONVENTIONS = {
    "return_convention": "daily simple returns of adjusted close, r = P_t/P_(t-1) - 1 (Stage 2 {ticker}_return)",
    "rebalancing": "daily constant mix: weights reset to target each trading day; no costs, taxes, financing or cash flows",
    "loss_sign_convention": "losses positive for VaR, CVaR, cumulative loss and limits; max drawdown reported negative",
    "units": "decimal fractions unless the field ends in _usd",
    "var_estimator": VAR_ESTIMATOR,
    "cvar_estimator": CVAR_ESTIMATOR,
    "volatility_estimator": "sample std (ddof=1); annualised by sqrt(annualization_days)",
}


def window_bounds(config: ProjectConfig, name: str) -> tuple[str | None, str | None]:
    """Inclusive date bounds of a named window (a Stage 2 split or ``full_sample``)."""
    if name == "full_sample":
        return None, None
    return config.data.splits.as_dict()[name]


def window_role(config: ProjectConfig, name: str) -> str:
    """How results from a window may be interpreted."""
    if name == config.risk.estimation_window:
        return "estimation window (calibration)"
    if name == "full_sample":
        return "descriptive only: spans train, validation and test; not out-of-sample evidence"
    return "evaluation window"


def _check_targets(output_dir: Path, overwrite: bool) -> None:
    existing = [f for f in OUTPUT_FILES if (output_dir / f).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains {existing}; pass overwrite=True (--overwrite) to replace them"
        )


def window_metrics(
    config: ProjectConfig, portfolio: Portfolio, asset_returns: pd.DataFrame, name: str
) -> dict[str, Any]:
    """Volatility, VaR, CVaR and drawdown of the portfolio over one window."""
    risk = config.risk
    start, end = window_bounds(config, name)
    aligned = apply_missing_data_policy(select_window(asset_returns, start, end),
                                        config.portfolio.missing_data_policy)
    r_p = calculate_portfolio_returns(aligned.returns, portfolio.weights)
    base = base_date_for(pd.DatetimeIndex(asset_returns.index), r_p.index[0])
    values = calculate_portfolio_value(r_p, portfolio.initial_capital, base)
    common = dict(capital=portfolio.initial_capital, min_observations=risk.min_observations,
                  min_tail_observations=risk.min_tail_observations, window=name)
    return {
        "window": name,
        "role": window_role(config, name),
        "configured_bounds": {"start": start, "end": end},
        "data": aligned.summary(),
        "base_date": base.date().isoformat(),
        "value_consistency_max_error": value_consistency_error(values, r_p),
        "volatility": calculate_historical_volatility(r_p, risk.annualization_days).to_dict(),
        "var": [calculate_historical_var(r_p, a, **common).to_dict() for a in risk.confidence_levels],
        "cvar": [calculate_historical_cvar(r_p, a, **common).to_dict() for a in risk.confidence_levels],
        "max_drawdown": calculate_max_drawdown(values).to_dict(),
        "final_value": float(values.iloc[-1]),
        "cumulative_return": float(values.iloc[-1] / values.iloc[0] - 1.0),
    }


def var_cvar_observations(metrics: dict[str, Any]) -> list[MetricObservation]:
    """1-day VaR/CVaR observations from one window's metrics, for limit checks."""
    obs = []
    for kind in ("var", "cvar"):
        for res in metrics[kind]:
            obs.append(MetricObservation(
                metric=f"historical_{kind}", value=res[f"{kind}_pct"],
                units="fraction_of_portfolio_value", horizon_days=1, confidence=res["confidence"],
                start_date=res["start_date"], end_date=res["end_date"], source=f"window:{metrics['window']}",
            ))
    return obs


def run_stress(
    config: ProjectConfig, portfolio: Portfolio, asset_returns: pd.DataFrame
) -> tuple[list[StressResult], pd.DataFrame]:
    """Configured direct shocks, configured replays and the worst windows in the estimation window."""
    risk = config.risk
    tol = risk.comparison_tolerance
    results = [
        apply_asset_return_shocks(portfolio, sc.shocks, sc.horizon_days, risk.limits,
                                  sc.name, sc.description, tol)
        for sc in risk.stress_scenarios
    ]
    results += [
        replay_historical_scenario(portfolio, asset_returns, rp.start_date, rp.horizon_days,
                                   risk.limits, rp.name, rp.description, tol)
        for rp in risk.historical_replays
    ]
    start, end = window_bounds(config, risk.estimation_window)
    est = select_window(asset_returns, start, end)
    worst = worst_historical_windows(est, portfolio.weights, risk.stress_horizon_days, risk.worst_windows)
    for i, row in enumerate(worst.itertuples(), start=1):
        results.append(replay_historical_scenario(
            portfolio, asset_returns, row.start_date, risk.stress_horizon_days, risk.limits,
            f"worst_{risk.stress_horizon_days}d_{risk.estimation_window}_{i}",
            f"Rank {i} worst non-overlapping {risk.stress_horizon_days}-day window in the "
            f"{risk.estimation_window} window (descriptive selection)", tol,
        ))
    return results, worst


def full_sample_tables(
    config: ProjectConfig, portfolio: Portfolio, asset_returns: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Portfolio returns (with split label, rolling vol, contributions) and value paths."""
    aligned = apply_missing_data_policy(asset_returns, config.portfolio.missing_data_policy)
    r_p = calculate_portfolio_returns(aligned.returns, portfolio.weights)
    split = pd.Series(None, index=r_p.index, dtype="object")
    for name, (s, e) in config.data.splits.as_dict().items():
        split[(r_p.index >= pd.Timestamp(s)) & (r_p.index <= pd.Timestamp(e))] = name
    contrib = return_contributions(aligned.returns, portfolio.weights).add_prefix("contrib_")
    returns_table = pd.concat(
        [split.rename("split"), r_p,
         rolling_volatility_table(r_p, config.risk.volatility_windows, config.risk.annualization_days),
         contrib], axis=1)
    base = base_date_for(pd.DatetimeIndex(asset_returns.index), r_p.index[0])
    values = calculate_portfolio_value(r_p, portfolio.initial_capital, base)
    bh = buy_and_hold_value(aligned.returns, portfolio.weights, portfolio.initial_capital, base)
    value_table = pd.DataFrame({
        "portfolio_value_constant_mix": values,
        "drawdown_constant_mix": drawdown_series(values),
        "portfolio_value_buy_and_hold": bh["total"],
    })
    value_table.index.name = "date"
    returns_table.index.name = "date"
    return returns_table, value_table


def run_risk_engine(
    config: ProjectConfig,
    source: pd.DataFrame,
    output_dir: Path,
    source_label: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run every Stage 3 calculation and write the outputs listed in :data:`OUTPUT_FILES`.

    Args:
        config: Validated project configuration.
        source: Stage 2 consolidated daily dataset (date index, ``{ticker}_return`` columns).
        output_dir: Directory for results.
        source_label: Where ``source`` came from (recorded in every output).
        overwrite: Replace existing Stage 3 outputs in ``output_dir``.

    Returns:
        Summary with metrics, stress results and output paths.
    """
    output_dir = Path(output_dir)
    _check_targets(output_dir, overwrite)
    universe = [a.alias for a in config.data.enabled_assets("stock")]
    portfolio = Portfolio.from_config(config.portfolio, universe=universe)
    asset_returns = calculate_asset_returns(source, portfolio.tickers, config.portfolio.return_column_template)
    risk = config.risk

    windows = list(dict.fromkeys([risk.estimation_window, *risk.report_windows]))
    metrics = {w: window_metrics(config, portfolio, asset_returns, w) for w in windows}
    est = metrics[risk.estimation_window]
    hist_limits = [l for l in risk.limits if l.metric.startswith("historical_")]
    var_limit_results = evaluate_risk_limits(hist_limits, var_cvar_observations(est), risk.comparison_tolerance)

    start, end = window_bounds(config, risk.estimation_window)
    est_aligned = apply_missing_data_policy(select_window(asset_returns, start, end),
                                            config.portfolio.missing_data_policy)
    contributions = calculate_asset_risk_contributions(
        est_aligned.returns, portfolio.weights, risk.annualization_days, risk.extreme_loss_dates)
    stress, worst = run_stress(config, portfolio, asset_returns)
    returns_table, value_table = full_sample_tables(config, portfolio, asset_returns)

    metadata = {
        "generated_at_utc": utc_now_iso(),
        "data_source": source_label,
        "data_date_range": {"start": source.index.min().date().isoformat(),
                            "end": source.index.max().date().isoformat(), "n_rows": int(len(source))},
        "portfolio": portfolio.describe(),
        "missing_data_policy": config.portfolio.missing_data_policy,
        "estimation_window": {"name": risk.estimation_window, "bounds": window_bounds(config, risk.estimation_window)},
        "splits": {k: list(v) for k, v in config.data.splits.as_dict().items()},
        "confidence_levels": list(risk.confidence_levels),
        "annualization_days": risk.annualization_days,
        "stress_horizon_days": risk.stress_horizon_days,
        "risk_limits": [asdict(l) for l in risk.limits],
        **CONVENTIONS,
    }
    vc = contributions["volatility_contributions"]
    historical = {
        "metadata": metadata,
        "windows": metrics,
        "historical_limit_results": [r.to_dict() for r in var_limit_results],
        "volatility_contributions": vc.summary(),
        "return_contribution_reconciliation_max_error": contributions["reconciliation_error"],
    }
    stress_payload = {
        "metadata": metadata,
        "worst_windows_searched_in": risk.estimation_window,
        "worst_windows": worst.to_dict(orient="records"),
        "scenarios": [s.to_dict() for s in stress],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "portfolio_returns": write_frame(returns_table, output_dir / "portfolio_returns.csv"),
        "portfolio_value": write_frame(value_table, output_dir / "portfolio_value.csv"),
        "historical_risk_metrics": write_json(historical, output_dir / "historical_risk_metrics.json"),
        "asset_risk_contributions": write_frame(contributions["summary"], output_dir / "asset_risk_contributions.csv"),
        "asset_return_correlation": write_frame(contributions["correlation"].rename_axis("ticker"),
                                                output_dir / "asset_return_correlation.csv"),
        "extreme_loss_contributions": write_frame(contributions["extreme_losses"],
                                                  output_dir / "extreme_loss_contributions.csv"),
        "stress_scenario_results": write_json(stress_payload, output_dir / "stress_scenario_results.json"),
    }
    report_path = output_dir / "risk_engine_report.md"
    report_path.write_text(render_report(metadata, metrics, var_limit_results, contributions, stress))
    out["risk_engine_report"] = report_path
    logger.info("Stage 3 outputs written to %s", output_dir)
    return {"metrics": metrics, "stress": stress, "limit_results": var_limit_results,
            "contributions": contributions, "outputs": {k: str(v) for k, v in out.items()},
            "metadata": metadata}


def _pct(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def _usd(x: float) -> str:
    return f"${x:,.0f}"


def render_report(
    metadata: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    limit_results: list[Any],
    contributions: dict[str, Any],
    stress: list[StressResult],
) -> str:
    """Human-readable Markdown summary of the Stage 3 run."""
    p = metadata["portfolio"]
    lines = [
        "# Stage 3 Risk Engine Report", "",
        f"Generated (UTC): {metadata['generated_at_utc']}  ",
        f"Data: `{metadata['data_source']}` ({metadata['data_date_range']['start']} to "
        f"{metadata['data_date_range']['end']}, {metadata['data_date_range']['n_rows']} rows)", "",
        f"Portfolio **{p['name']}**: {_usd(p['initial_capital'])} {p['currency']}, policy {p['policy']}, "
        f"weights " + ", ".join(f"{k} {v:.1%}" for k, v in p["weights"].items()), "",
        "Conventions: " + metadata["return_convention"] + "; " + metadata["rebalancing"] + "; "
        + metadata["loss_sign_convention"] + ".", "",
        "## Historical risk metrics (1-day)", "",
        "| Window | Role | Obs | Dates | Vol daily | Vol ann. | " +
        " | ".join(f"VaR {a:.0%} | CVaR {a:.0%}" for a in metadata["confidence_levels"]) + " | Max DD |",
        "|---|---|---|---|---|---|" + "---|---|" * len(metadata["confidence_levels"]) + "---|",
    ]
    for name, m in metrics.items():
        cells = [name, m["role"], str(m["volatility"]["n_observations"]),
                 f"{m['data']['start_date']} to {m['data']['end_date']}",
                 _pct(m["volatility"]["daily_std"], 3), _pct(m["volatility"]["annualized"])]
        for v, c in zip(m["var"], m["cvar"]):
            cells += [f"{_pct(v['var_pct'])} ({_usd(v['var_usd'])})", f"{_pct(c['cvar_pct'])} ({_usd(c['cvar_usd'])})"]
        dd = m["max_drawdown"]
        cells.append(f"{_pct(dd['max_drawdown'])} ({dd['peak_date']} to {dd['trough_date']}, "
                     f"recovered {dd['recovery_date'] or 'not in window'})")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Excluded dates (missing holding returns) per window: " + "; ".join(
        f"{n}: {m['data']['n_excluded_rows']}" for n, m in metrics.items()), ""]
    lines += ["## Historical limit checks (estimation window)", "",
              "| Limit | Metric | Threshold | Observed | Status |", "|---|---|---|---|---|"]
    for r in limit_results:
        lines.append(f"| {r.limit_name} | {r.metric} {r.confidence or ''} {r.horizon_days}d | "
                     f"{r.breach_rule} {_pct(r.threshold)} | {_pct(r.observed)} | {r.status} |")
    vc = contributions["volatility_contributions"]
    lines += ["", f"## Asset diagnostics ({vc.start_date} to {vc.end_date}, {vc.n_observations} obs)", "",
              f"Portfolio volatility {_pct(vc.portfolio_vol_daily, 3)} daily / {_pct(vc.portfolio_vol_annualized)} "
              f"annualised. Method: {vc.method}.", "",
              "| Ticker | Weight | Asset vol ann. | Vol contribution ann. | Share of portfolio vol | Mean daily return contribution |",
              "|---|---|---|---|---|---|"]
    for t, row in contributions["summary"].iterrows():
        lines.append(f"| {t} | {row['weight']:.1%} | {_pct(row['asset_vol_annualized'])} | "
                     f"{_pct(row['vol_contribution_annualized'])} | {row['vol_contribution_share']:.1%} | "
                     f"{_pct(row['mean_return_contribution_daily'], 4)} |")
    lines += ["", "## Stress scenarios", "",
              "| Scenario | Type | Horizon | Dates | Portfolio return | Loss | Path max DD | Limits |",
              "|---|---|---|---|---|---|---|---|"]
    for s in stress:
        lim = ", ".join(f"{r.limit_name}: {r.status}" for r in s.limit_results)
        dates = f"{s.start_date} to {s.end_date}" if s.start_date else "hypothetical"
        lines.append(f"| {s.scenario_name} | {s.scenario_type} | {s.horizon_days}d | {dates} | "
                     f"{_pct(s.portfolio_return)} | {_usd(s.loss_usd)} ({_pct(s.loss_pct)}) | "
                     f"{_pct(s.max_drawdown)} | {lim} |")
    lines += ["", "## Limitations", "",
              "- Historical VaR/CVaR describe the chosen window only; they are not forecasts or maximum losses.",
              "- Direct shocks are hypothetical inputs; historical replays cover only episodes in the data.",
              "- Constant-mix assumes free daily rebalancing; no costs, taxes, financing or cash flows.",
              "- Full-sample figures span the test split and are descriptive only.", ""]
    return "\n".join(lines)
