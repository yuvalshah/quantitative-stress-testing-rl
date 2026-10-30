"""Integration: Stage 2 pipeline on synthetic sources -> Stage 3 engine and demo script.

All data are synthetic (SYNA/SYNB tickers) and written to pytest temp dirs only.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rlstress.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, config_from_dict
from rlstress.data.pipeline import run_pipeline
from rlstress.data.storage import read_frame
from rlstress.risk_report import OUTPUT_FILES

from conftest import SYNTHETIC_DATA_SECTION


def _load_demo():
    spec = importlib.util.spec_from_file_location("stage3_demo", PROJECT_ROOT / "examples" / "stage3_risk_engine_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def synthetic_project(tmp_path, data_config, fetchers):
    run_pipeline(data_config, root=tmp_path, fetchers=fetchers, plots=False)
    with DEFAULT_CONFIG_PATH.open("rb") as fh:
        raw = tomllib.load(fh)
    raw = copy.deepcopy(raw)
    raw["data"] = copy.deepcopy(SYNTHETIC_DATA_SECTION)
    raw["portfolio"]["name"] = "synthetic_ab"
    raw["portfolio"]["weights"] = {"SYNA": 0.6, "SYNB": 0.4}
    raw["risk"]["min_observations"] = 50
    raw["risk"]["report_windows"] = ["train", "validation", "full_sample"]
    raw["risk"]["stress_scenarios"] = [{"name": "hypo", "horizon_days": 5, "shocks": {"SYNA": -0.10, "SYNB": -0.05}}]
    raw["risk"]["historical_replays"] = [{"name": "replay", "start_date": "2020-04-01", "horizon_days": 5}]
    config = config_from_dict(raw)
    returns_path = tmp_path / "data" / "processed" / "features" / "consolidated_daily.csv"
    return config, returns_path, tmp_path


def test_demo_end_to_end(synthetic_project) -> None:
    config, returns_path, tmp = synthetic_project
    out_dir = tmp / "results" / "stage3"
    summary = _load_demo().run_demo(config, returns_path, out_dir)
    for f in OUTPUT_FILES:
        assert (out_dir / f).exists(), f

    hist = json.loads((out_dir / "historical_risk_metrics.json").read_text())
    meta = hist["metadata"]
    for key in ("portfolio", "data_date_range", "estimation_window", "return_convention", "rebalancing",
                "confidence_levels", "loss_sign_convention", "stress_horizon_days", "risk_limits",
                "missing_data_policy", "data_source"):
        assert key in meta, key
    assert meta["data_source"] == str(returns_path)
    train = hist["windows"]["train"]
    assert train["role"].startswith("estimation window")
    assert hist["windows"]["full_sample"]["role"].startswith("descriptive")
    assert train["data"]["end_date"] <= "2020-06-30"
    assert train["value_consistency_max_error"] < 1e-9
    # SYNB lacks one day: that day and the next have no SYNB return -> excluded, plus the first row
    assert hist["windows"]["full_sample"]["data"]["n_excluded_rows"] == 3

    # recompute 95% VaR from the written portfolio returns
    pr = read_frame(out_dir / "portfolio_returns.csv")
    tr = pr.loc[:"2020-06-30", "portfolio_return"].to_numpy()
    expected = -np.quantile(tr, 0.05, method="inverted_cdf")
    assert train["var"][0]["var_pct"] == pytest.approx(expected)
    assert {r["limit_name"] for r in hist["historical_limit_results"]} == {"historical_var_95_1d", "historical_cvar_95_1d"}

    stress = json.loads((out_dir / "stress_scenario_results.json").read_text())
    hypo = next(s for s in stress["scenarios"] if s["scenario_name"] == "hypo")
    assert hypo["loss_usd"] == pytest.approx(100_000 * 0.08)
    assert hypo["breached"] is False
    assert any(s["scenario_type"] == "historical_replay" for s in stress["scenarios"])
    assert len(stress["worst_windows"]) >= 1
    assert "Stage 3 Risk Engine Report" in (out_dir / "risk_engine_report.md").read_text()
    assert summary["contributions"]["reconciliation_error"] < 1e-12


def test_demo_refuses_to_overwrite(synthetic_project) -> None:
    config, returns_path, tmp = synthetic_project
    demo = _load_demo()
    out_dir = tmp / "out"
    demo.run_demo(config, returns_path, out_dir)
    with pytest.raises(FileExistsError, match="overwrite"):
        demo.run_demo(config, returns_path, out_dir)
    demo.run_demo(config, returns_path, out_dir, overwrite=True)


def test_missing_returns_file_reported(synthetic_project) -> None:
    config, _, tmp = synthetic_project
    with pytest.raises(FileNotFoundError, match="Stage 2"):
        _load_demo().run_demo(config, tmp / "nope.csv", tmp / "out")


def test_engine_does_not_mutate_source(synthetic_project) -> None:
    from rlstress.risk_report import run_risk_engine

    config, returns_path, tmp = synthetic_project
    source = read_frame(returns_path)
    before = source.copy()
    run_risk_engine(config, source, tmp / "o", "synthetic")
    pd.testing.assert_frame_equal(source, before)
