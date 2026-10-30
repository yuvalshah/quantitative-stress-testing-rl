"""Shared fixtures: a synthetic data config and offline fake data sources.

All synthetic series use made-up identifiers (``SYNA``, ``^SYNVOL``, ``SYNYLD``...)
so they cannot be mistaken for real market data, and are only written to
pytest's temporary directories.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from rlstress.config import DataConfig, data_config_from_dict, validate_data_config
from rlstress.data.pipeline import Fetchers

SYNTHETIC_DATA_SECTION: dict[str, Any] = {
    "start_date": "2020-01-01",
    "end_date": "2020-12-31",
    "seed": 42,
    "paths": {"raw_dir": "data/raw", "processed_dir": "data/processed", "reports_dir": "data/reports"},
    "yahoo": {"max_retries": 2, "retry_backoff_seconds": 0.0, "pause_seconds": 0.0},
    "fred": {"max_retries": 2, "retry_backoff_seconds": 0.0},
    "features": {
        "windows": [3, 5],
        "correlation_windows": [5],
        "drawdown_windows": [5],
        "min_obs_fraction": 0.8,
        "regime_source": "SYNVOL_close",
        "correlation_pairs": {"SYNA": ["SYNM", "SYND"], "SYNB": ["SYNM"]},
    },
    "splits": {
        "train": ["2020-01-01", "2020-06-30"],
        "validation": ["2020-07-01", "2020-09-30"],
        "test": ["2020-10-01", "2020-12-31"],
    },
    "assets": [
        {"symbol": "SYNA", "alias": "SYNA", "name": "Synthetic A", "group": "stock"},
        {"symbol": "SYNB", "alias": "SYNB", "name": "Synthetic B", "group": "stock"},
        {"symbol": "SYNM", "alias": "SYNM", "name": "Synthetic market", "group": "market_indicator"},
        {"symbol": "SYND", "alias": "SYND", "name": "Disabled", "group": "market_indicator",
         "required": False, "enabled": False},
        {"symbol": "^SYNVOL", "alias": "SYNVOL", "name": "Synthetic vol index", "group": "volatility"},
    ],
    "macro": [
        {"series_id": "SYNYLD", "alias": "SYNYLD", "name": "Synthetic yield", "kind": "yield", "unit": "percent"},
        {"series_id": "SYNOIL", "alias": "SYNOIL", "name": "Synthetic oil", "kind": "price", "unit": "usd_per_barrel"},
    ],
}

SYNB_MISSING_DAY = "2020-03-10"  # deliberately absent from SYNB to create a gap


@pytest.fixture()
def data_config() -> DataConfig:
    cfg = data_config_from_dict(json.loads(json.dumps(SYNTHETIC_DATA_SECTION)))
    validate_data_config(cfg)
    return cfg


def synthetic_yahoo_history(symbol: str, start: str, end_inclusive: str, settings: Any) -> pd.DataFrame:
    """Deterministic yfinance-shaped response with a tz-aware index."""
    seed = 42 + sum(map(ord, symbol))
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end_inclusive)
    if symbol == "SYNB":
        days = days[days != pd.Timestamp(SYNB_MISSING_DAY)]
    tz = "America/Chicago" if symbol.startswith("^") else "America/New_York"
    idx = pd.DatetimeIndex(days).tz_localize(tz)
    base = 20.0 if symbol.startswith("^") else 100.0
    close = base * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Adj Close": close * 0.98,
            "Volume": rng.integers(1_000, 10_000, len(idx)),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=pd.Index(idx, name="Date"),
    )


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        return self._payload


def synthetic_fred_get(url: str, params: dict[str, Any], timeout: float) -> FakeResponse:
    """Fake FRED API: business-day observations with a holiday marker ``"."``."""
    sid = params["series_id"]
    if url.endswith("/series"):
        return FakeResponse(payload={"seriess": [{"id": sid, "units": "Percent", "frequency": "Daily"}]})
    days = pd.bdate_range(params["observation_start"], params["observation_end"])
    rng = np.random.default_rng(7 + len(sid))
    level = (1.5 if sid == "SYNYLD" else 40.0) + np.cumsum(rng.normal(0, 0.02, len(days)))
    obs = [{"date": d.date().isoformat(), "value": f"{v:.2f}"} for d, v in zip(days, level)]
    obs[10]["value"] = "."  # FRED "no observation" marker
    return FakeResponse(payload={"observations": obs})


@pytest.fixture()
def fetchers() -> Fetchers:
    return Fetchers(
        yahoo=synthetic_yahoo_history,
        http_get=synthetic_fred_get,
        environ={"FRED_API_KEY": "test-key-not-real"},
        sleep=lambda _s: None,
    )


# --- Stage 4 fixtures: synthetic factor world with known loadings -----------------------------

STAGE4_FACTORS = ("SPY_return", "QQQ_return", "XLF_return", "XLE_return")
# True raw loadings r = a + B f + e (rows SYNA "tech-like", SYNB "financial-like")
STAGE4_B = np.array([[0.3, 0.9, 0.0, 0.0],
                     [0.2, 0.0, 1.1, 0.1]])
STAGE4_A = np.array([0.0002, 0.0001])


def stage4_frame(seed: int = 42) -> pd.DataFrame:
    """Synthetic consolidated dataset 2015-2020 with fat-tailed, correlated factors (SYN tickers)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", "2020-12-31", name="date")
    n = len(idx)
    corr = np.array([[1.0, 0.9, 0.8, 0.6], [0.9, 1.0, 0.7, 0.5], [0.8, 0.7, 1.0, 0.5], [0.6, 0.5, 0.5, 1.0]])
    sd = np.array([0.010, 0.012, 0.013, 0.018])
    cov_f = corr * np.outer(sd, sd)
    cov_e = np.array([[1.0, 0.3], [0.3, 1.0]]) * 0.008 ** 2
    cov = np.block([[cov_f, np.zeros((4, 2))], [np.zeros((2, 4)), cov_e]])
    nu = 5.0
    z = rng.multivariate_normal(np.zeros(6), cov * (nu - 2) / nu, size=n)
    w = rng.chisquare(nu, size=n) / nu
    x = z / np.sqrt(w)[:, None]
    f, e = x[:, :4], x[:, 4:]
    r = STAGE4_A + f @ STAGE4_B.T + e
    frame = pd.DataFrame(f, index=idx, columns=list(STAGE4_FACTORS))
    frame["SYNA_return"] = r[:, 0]
    frame["SYNB_return"] = r[:, 1]
    frame.iloc[0] = np.nan  # first row has no returns, as in Stage 2 output
    frame["VIX_close"] = 15 + 2 * np.abs(rng.standard_normal(n))
    frame["SPY_vol_ann_20d"] = frame["SPY_return"].rolling(20, min_periods=16).std() * np.sqrt(252)
    return frame


def stage4_raw_config(**risk_overrides) -> dict:
    import copy
    import tomllib

    from rlstress.config import DEFAULT_CONFIG_PATH

    with DEFAULT_CONFIG_PATH.open("rb") as fh:
        raw = copy.deepcopy(tomllib.load(fh))
    data = copy.deepcopy(SYNTHETIC_DATA_SECTION)
    data["start_date"], data["end_date"] = "2015-01-01", "2020-12-31"
    data["splits"] = {"train": ["2015-01-01", "2018-12-31"], "validation": ["2019-01-01", "2019-12-31"],
                      "test": ["2020-01-01", "2020-12-31"]}
    raw["data"] = data
    raw["portfolio"]["name"] = "synthetic_ab"
    raw["portfolio"]["weights"] = {"SYNA": 0.5, "SYNB": 0.5}
    raw["risk"]["stress_scenarios"] = [{"name": "hypo", "horizon_days": 5, "shocks": {"SYNA": -0.1, "SYNB": -0.05}}]
    raw["risk"]["historical_replays"] = []
    raw["risk"].update(risk_overrides)
    raw["simulation"]["dof_grid"] = [3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 20.0]
    raw["simulation"]["diagnostic_simulations"] = 5000
    return raw


@pytest.fixture()
def s4_source() -> pd.DataFrame:
    return stage4_frame()


@pytest.fixture()
def s4_config():
    from rlstress.config import config_from_dict

    return config_from_dict(stage4_raw_config())
