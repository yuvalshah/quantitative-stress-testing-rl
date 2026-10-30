"""Tests for configuration loading and validation."""

from __future__ import annotations

import copy
import tomllib

import pytest

from rlstress.config import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    ProjectConfig,
    config_from_dict,
    load_config,
)


@pytest.fixture()
def raw_config() -> dict:
    with DEFAULT_CONFIG_PATH.open("rb") as fh:
        return tomllib.load(fh)


def test_default_config_loads() -> None:
    config = load_config()
    assert isinstance(config, ProjectConfig)
    assert len(config.portfolio.tickers) == len(config.portfolio.weights) == 8
    assert config.portfolio.initial_capital == 100_000.0
    assert config.risk.confidence_levels == (0.95, 0.99)
    assert {l.name for l in config.risk.limits} >= {"stress_loss_5d"}


def test_weight_sum_checked_when_portfolio_is_built(raw_config: dict) -> None:
    from rlstress.exceptions import PortfolioValidationError
    from rlstress.portfolio import Portfolio

    bad = copy.deepcopy(raw_config)
    bad["portfolio"]["weights"]["AAPL"] += 0.1
    cfg = config_from_dict(bad)  # structurally valid
    with pytest.raises(PortfolioValidationError, match="sum"):
        Portfolio.from_config(cfg.portfolio)


def test_non_numeric_weight_rejected(raw_config: dict) -> None:
    bad = copy.deepcopy(raw_config)
    bad["portfolio"]["weights"]["AAPL"] = "0.125"
    with pytest.raises(ConfigError, match="numbers"):
        config_from_dict(bad)


@pytest.mark.parametrize("capital", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_capital_rejected(raw_config: dict, capital: float) -> None:
    bad = copy.deepcopy(raw_config)
    bad["portfolio"]["initial_capital"] = capital
    with pytest.raises(ConfigError, match="initial_capital"):
        config_from_dict(bad)


def test_missing_section_is_reported(raw_config: dict) -> None:
    bad = copy.deepcopy(raw_config)
    del bad["risk"]
    with pytest.raises(ConfigError, match="risk"):
        config_from_dict(bad)


def test_unknown_field_is_reported(raw_config: dict) -> None:
    bad = copy.deepcopy(raw_config)
    bad["simulation"]["unknown_param"] = 1
    with pytest.raises(ConfigError, match="simulation"):
        config_from_dict(bad)


@pytest.mark.parametrize("alpha", [1.5, 0.0, 1.0, -0.2])
def test_invalid_confidence_levels(raw_config: dict, alpha: float) -> None:
    bad = copy.deepcopy(raw_config)
    bad["risk"]["confidence_levels"] = [0.95, alpha]
    with pytest.raises(ConfigError, match="alpha"):
        config_from_dict(bad)


@pytest.mark.parametrize("field,value,match", [
    ("units", "percent", "units"),
    ("metric", "parametric_var", "metric"),
    ("horizon_days", 0, "horizon_days"),
    ("threshold", 10.0, "threshold"),
    ("breach_rule", "=>", "breach_rule"),
])
def test_invalid_limit_rejected(raw_config: dict, field: str, value, match: str) -> None:
    bad = copy.deepcopy(raw_config)
    bad["risk"]["limits"][0][field] = value
    with pytest.raises(ConfigError, match=match):
        config_from_dict(bad)


def test_var_limit_needs_confidence_and_daily_horizon(raw_config: dict) -> None:
    bad = copy.deepcopy(raw_config)
    var_limit = next(l for l in bad["risk"]["limits"] if l["metric"] == "historical_var")
    del var_limit["confidence"]
    with pytest.raises(ConfigError, match="confidence"):
        config_from_dict(bad)
    bad = copy.deepcopy(raw_config)
    var_limit = next(l for l in bad["risk"]["limits"] if l["metric"] == "historical_var")
    var_limit["horizon_days"] = 5
    with pytest.raises(ConfigError, match="horizon_days = 1"):
        config_from_dict(bad)


def test_unknown_estimation_window_rejected(raw_config: dict) -> None:
    bad = copy.deepcopy(raw_config)
    bad["risk"]["estimation_window"] = "everything"
    with pytest.raises(ConfigError, match="estimation_window"):
        config_from_dict(bad)


# --- Group A: Stage 2 data configuration ---------------------------------------------------

from rlstress.config import DataPaths, data_config_from_dict, validate_data_config  # noqa: E402
from rlstress.data.storage import ensure_directories  # noqa: E402


def _data(raw_config: dict) -> dict:
    return copy.deepcopy(raw_config["data"])


def _validate(data: dict) -> None:
    validate_data_config(data_config_from_dict(data))


def test_default_data_universe(raw_config: dict) -> None:
    data = load_config().data
    stocks = [a.alias for a in data.enabled_assets("stock")]
    assert stocks == ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "WMT", "CAT"]
    assert [a.alias for a in data.enabled_assets("market_indicator")] == ["SPY", "QQQ", "XLF", "XLE"]
    assert [a.symbol for a in data.enabled_assets("volatility")] == ["^VIX"]
    assert [m.series_id for m in data.enabled_macro()] == ["DGS10", "DFF", "DCOILWTICO"]
    assert (data.start_date, data.end_date) == ("2015-01-01", "2025-12-31")
    assert data.features.windows == (5, 20, 60)


@pytest.mark.parametrize("start,end", [("2015-13-01", "2025-12-31"), ("2025-01-01", "2015-01-01"),
                                       ("yesterday", "2025-12-31")])
def test_invalid_dates_rejected(raw_config: dict, start: str, end: str) -> None:
    data = _data(raw_config)
    data["start_date"], data["end_date"] = start, end
    with pytest.raises(ConfigError):
        _validate(data)


@pytest.mark.parametrize("field,value", [("symbol", "aapl lower"), ("alias", "bad-alias"),
                                         ("group", "crypto")])
def test_invalid_asset_rejected(raw_config: dict, field: str, value: str) -> None:
    data = _data(raw_config)
    data["assets"][0][field] = value
    with pytest.raises(ConfigError):
        _validate(data)


def test_invalid_series_id_and_duplicate_alias(raw_config: dict) -> None:
    data = _data(raw_config)
    data["macro"][0]["series_id"] = "dgs-10"
    with pytest.raises(ConfigError, match="series id"):
        _validate(data)
    data = _data(raw_config)
    data["assets"][1]["alias"] = data["assets"][0]["alias"]
    with pytest.raises(ConfigError, match="Duplicate"):
        _validate(data)


def test_unknown_correlation_alias_rejected(raw_config: dict) -> None:
    data = _data(raw_config)
    data["features"]["correlation_pairs"]["AAPL"] = ["NOPE"]
    with pytest.raises(ConfigError, match="NOPE"):
        _validate(data)


def test_windows_configurable_and_validated(raw_config: dict) -> None:
    data = _data(raw_config)
    data["features"]["windows"] = [10, 30]
    cfg = data_config_from_dict(data)
    validate_data_config(cfg)
    assert cfg.features.windows == (10, 30)
    assert cfg.features.min_periods(10) == 8
    data["features"]["windows"] = [1]
    with pytest.raises(ConfigError, match="windows"):
        _validate(data)


@pytest.mark.parametrize("split,value", [("validation", ["2021-06-01", "2023-12-31"]),
                                         ("test", ["2024-01-01", "2026-06-30"])])
def test_bad_split_boundaries_rejected(raw_config: dict, split: str, value: list[str]) -> None:
    data = _data(raw_config)
    data["splits"][split] = value
    with pytest.raises(ConfigError, match="splits"):
        _validate(data)


def test_output_paths_resolved_and_created(tmp_path) -> None:
    paths = DataPaths().resolve(tmp_path)
    dirs = ensure_directories(paths)
    for name in ("raw_stock", "raw_market_indicator", "raw_volatility", "raw_macro",
                 "prices", "returns", "features", "splits", "reports"):
        assert dirs[name].is_dir() and str(dirs[name]).startswith(str(tmp_path))
