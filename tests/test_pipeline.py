"""Test group H: end-to-end pipeline on synthetic offline sources (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rlstress.data.pipeline import Fetchers, run_pipeline
from rlstress.data.storage import read_frame

from conftest import SYNB_MISSING_DAY, synthetic_fred_get, synthetic_yahoo_history


@pytest.fixture()
def run(tmp_path, data_config, fetchers):
    return run_pipeline(data_config, root=tmp_path, fetchers=fetchers, plots=False)


def test_pipeline_produces_expected_outputs(run, tmp_path) -> None:
    assert run.exit_code == 0
    for key in ("consolidated_daily", "stock_returns", "macro_changes", "split_train",
                "split_validation", "split_test", "data_dictionary", "source_manifest",
                "quality_report_json", "quality_report_md", "standardizer", "regime_thresholds"):
        assert Path(run.outputs[key]).exists(), key
    for alias, group in [("SYNA", "stocks"), ("SYNM", "market_indicators"),
                         ("SYNVOL", "volatility"), ("SYNYLD", "macro")]:
        assert (tmp_path / "data" / "raw" / group / f"{alias}.csv").exists()
    assert not (tmp_path / "data" / "raw" / "market_indicators" / "SYND.csv").exists()  # disabled


def test_consolidated_dataset_semantics(run) -> None:
    df = read_frame(Path(run.outputs["consolidated_daily"]))
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert (df.index.dayofweek < 5).all()  # calendar = stock trading days only
    gap = pd.Timestamp(SYNB_MISSING_DAY)
    nxt = df.index[df.index.get_loc(gap) + 1]
    assert pd.isna(df.loc[gap, "SYNB_return"]) and pd.isna(df.loc[nxt, "SYNB_return"])
    assert pd.notna(df.loc[gap, "SYNA_return"])  # other stock unaffected
    for col in ("SYNA_vol_ann_5d", "SYNA_SYNM_corr_5d", "SYNA_drawdown_5d", "SYNVOL_mean_3d",
                "SYNYLD_yield_pct", "SYNYLD_change_bp", "SYNOIL_return", "SYNVOL_regime"):
        assert col in df.columns, col
    assert not any("SYND" in c for c in df.columns)
    assert not (df.filter(like="_return") == 0).all().any()


def test_returns_match_raw_prices(run) -> None:
    df = read_frame(Path(run.outputs["consolidated_daily"]))
    px = df["SYNA_adj_close"]
    manual = px.iloc[5] / px.iloc[4] - 1
    assert df["SYNA_return"].iloc[5] == pytest.approx(manual)


def test_split_files_partition_the_dataset(run) -> None:
    full = read_frame(Path(run.outputs["consolidated_daily"]))
    parts = [read_frame(Path(run.outputs[f"split_{n}"])) for n in ("train", "validation", "test")]
    assert sum(len(p) for p in parts) == len(full)
    assert parts[0].index.max() < parts[1].index.min() and parts[1].index.max() < parts[2].index.min()
    assert parts[0].index.max() <= pd.Timestamp("2020-06-30")


def test_regime_thresholds_from_training_only(run) -> None:
    regime = json.loads(Path(run.outputs["regime_thresholds"]).read_text())
    full = read_frame(Path(run.outputs["consolidated_daily"]))
    train = full.loc[:"2020-06-30", "SYNVOL_close"]
    assert regime["thresholds"][0] == pytest.approx(train.quantile(0.25))
    assert regime["thresholds"][0] != pytest.approx(full["SYNVOL_close"].quantile(0.25))


def test_report_and_dictionary_contents(run, tmp_path) -> None:
    report = json.loads(Path(run.outputs["quality_report_json"]).read_text())
    assert report["overall_assessment"] in ("PASSED", "PASSED WITH WARNINGS")
    assert report["stock_collection_status"] == "COMPLETE"
    names = {c["name"] for c in report["checks"] if c["status"] == "passed"}
    assert "no look-ahead in features" in names
    assert report["alignment"]["per_source"]["SYNB"]["n_calendar_dates_missing_in_source"] == 1
    assert report["cleaning"]["SYNYLD"]["source_missing_markers"] == 1
    assert "SYNA-SYND" in " ".join(report["warnings"])
    dictionary = pd.read_csv(run.outputs["data_dictionary"])
    required = {"dataset", "variable", "source", "source_id", "description", "unit",
                "original_frequency", "transformed_frequency", "transformation",
                "first_valid_date", "last_valid_date", "n_observations", "n_missing",
                "category", "data_kind", "limitations"}
    assert required <= set(dictionary.columns)
    row = dictionary[(dictionary.dataset == "consolidated_daily") & (dictionary.variable == "SYNYLD_change_bp")].iloc[0]
    assert row.unit == "basis points" and row.data_kind == "derived"
    manifest = json.loads(Path(run.outputs["source_manifest"]).read_text())
    assert all("retrieved_at_utc" in s for s in manifest["sources"])
    assert "test-key-not-real" not in Path(run.outputs["quality_report_json"]).read_text()
    assert "test-key-not-real" not in Path(run.outputs["source_manifest"]).read_text()


def test_rerun_uses_cache_and_does_not_duplicate(tmp_path, data_config, fetchers) -> None:
    first = run_pipeline(data_config, root=tmp_path, fetchers=fetchers, plots=False)
    calls = {"n": 0}

    def counting(*args):
        calls["n"] += 1
        return synthetic_yahoo_history(*args)

    second = run_pipeline(data_config, root=tmp_path, plots=False,
                          fetchers=Fetchers(yahoo=counting, http_get=synthetic_fred_get,
                                            environ={"FRED_API_KEY": "k"}, sleep=lambda s: None))
    assert calls["n"] == 0
    assert all(r.status == "cached" for r in second.results)
    raw = pd.read_csv(tmp_path / "data" / "raw" / "stocks" / "SYNA.csv")
    assert raw["date"].is_unique
    pd.testing.assert_frame_equal(read_frame(Path(first.outputs["consolidated_daily"])),
                                  read_frame(Path(second.outputs["consolidated_daily"])))


def test_force_refresh_archives_raw(tmp_path, data_config, fetchers) -> None:
    run_pipeline(data_config, root=tmp_path, fetchers=fetchers, plots=False)
    run_pipeline(data_config, root=tmp_path, fetchers=fetchers, plots=False, force=True)
    archive = tmp_path / "data" / "raw" / "stocks" / "_archive"
    assert any(p.name.startswith("SYNA_") for p in archive.iterdir())
    assert pd.read_csv(tmp_path / "data" / "raw" / "stocks" / "SYNA.csv")["date"].is_unique


def test_optional_macro_failure_does_not_stop_pipeline(tmp_path, data_config) -> None:
    fetchers = Fetchers(yahoo=synthetic_yahoo_history, http_get=synthetic_fred_get,
                        environ={}, sleep=lambda s: None)  # no FRED key
    run = run_pipeline(data_config, root=tmp_path, fetchers=fetchers, plots=False)
    assert run.exit_code == 0
    statuses = {r.alias: r.status for r in run.results}
    assert statuses["SYNYLD"] == "failed" and statuses["SYNA"] == "success"
    warn = [c for c in run.report["checks"] if c["scope"] == "macro:SYNYLD"]
    assert warn and warn[0]["status"] == "warning"
    assert "consolidated_daily" in run.outputs


def test_required_stock_failure_sets_exit_code(tmp_path, data_config) -> None:
    def no_synb(symbol, *args):
        if symbol == "SYNB":
            return pd.DataFrame()
        return synthetic_yahoo_history(symbol, *args)

    run = run_pipeline(data_config, root=tmp_path, plots=False,
                       fetchers=Fetchers(yahoo=no_synb, http_get=synthetic_fred_get,
                                         environ={"FRED_API_KEY": "k"}, sleep=lambda s: None))
    assert run.exit_code == 1
    assert run.report["stock_collection_status"].startswith("INCOMPLETE")
    assert run.report["overall_assessment"] == "FAILED"


def test_offline_without_cache_fails_honestly(tmp_path, data_config, fetchers) -> None:
    run = run_pipeline(data_config, root=tmp_path, fetchers=fetchers, plots=False, offline=True)
    assert run.exit_code == 1
    assert all(r.status == "failed" for r in run.results)
    assert "consolidated_daily" not in run.outputs


def test_plots_written_when_data_available(tmp_path, data_config, fetchers) -> None:
    pytest.importorskip("matplotlib")
    run = run_pipeline(data_config, root=tmp_path, fetchers=fetchers, plots=True)
    figures = run.report["figures"]
    assert {"stock_adjusted_prices", "stock_return_distributions", "coverage"} <= set(figures)
    assert "wti" not in figures  # no WTI series in the synthetic config -> skipped, not faked
    assert all(Path(p).stat().st_size > 0 for p in figures.values())
