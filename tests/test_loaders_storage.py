"""Loaders (mocked network) and the raw storage layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlstress.config import FredSettings, MacroSeriesSpec, YahooSettings, AssetSpec
from rlstress.data.fred_loader import download_fred_series, parse_fred_csv, resolve_fred_api_key
from rlstress.data.sources import (
    DownloadError,
    TransientSourceError,
    assess_coverage,
    exclusive_end,
    with_retries,
)
from rlstress.data.storage import (
    cache_is_valid,
    raw_file_path,
    read_raw_series,
    write_raw_series,
)
from rlstress.data.yahoo_loader import download_yahoo_asset, normalize_yahoo_frame

from conftest import FakeResponse, synthetic_yahoo_history

SPEC = MacroSeriesSpec("SYNYLD", "SYNYLD", "Synthetic yield", "yield", "percent")


def test_exclusive_end_adds_one_day() -> None:
    assert exclusive_end("2025-12-31") == "2026-01-01"


def test_normalize_keeps_local_trading_date() -> None:
    raw = synthetic_yahoo_history("^SYNVOL", "2024-01-02", "2024-01-05", YahooSettings())
    out = normalize_yahoo_frame(raw, "^SYNVOL")
    assert list(out.columns) == ["date", "open", "high", "low", "close", "adj_close",
                                 "volume", "dividends", "stock_splits"]
    assert out["date"].dt.tz is None
    assert out["date"].iloc[0] == pd.Timestamp("2024-01-02")
    np.testing.assert_allclose(out["adj_close"], raw["Adj Close"])  # values untouched


def test_normalize_handles_multiindex_columns() -> None:
    raw = synthetic_yahoo_history("SYNA", "2024-01-02", "2024-01-03", YahooSettings())
    multi = raw.copy()
    multi.columns = pd.MultiIndex.from_product([raw.columns, ["SYNA"]], names=["Price", "Ticker"])
    pd.testing.assert_frame_equal(normalize_yahoo_frame(multi, "SYNA"), normalize_yahoo_frame(raw, "SYNA"))


def test_yahoo_download_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def flaky(symbol, start, end, settings):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TransientSourceError("rate limited")
        return synthetic_yahoo_history(symbol, start, end, settings)

    spec = AssetSpec("SYNA", "SYNA", "A", "stock")
    r = download_yahoo_asset(spec, "2024-01-01", "2024-01-31", YahooSettings(max_retries=3), 10,
                             fetcher=flaky, sleep=lambda s: None)
    assert r.status == "success" and calls["n"] == 2 and r.n_records > 0


def test_yahoo_empty_response_fails_cleanly() -> None:
    spec = AssetSpec("SYNA", "SYNA", "A", "stock")
    r = download_yahoo_asset(spec, "2024-01-01", "2024-01-31", YahooSettings(max_retries=2), 10,
                             fetcher=lambda *a: pd.DataFrame(), sleep=lambda s: None)
    assert r.status == "failed" and "no rows" in r.message


def test_with_retries_raises_download_error() -> None:
    def boom():
        raise TransientSourceError("x")

    with pytest.raises(DownloadError, match="2 attempt"):
        with_retries(boom, attempts=2, backoff_seconds=0, description="t", sleep=lambda s: None)


def test_coverage_warning_for_late_start() -> None:
    assert assess_coverage("2015-03-01", "2025-12-31", "2015-01-01", "2025-12-31", 10)
    assert assess_coverage("2015-01-02", "2025-12-31", "2015-01-01", "2025-12-31", 10) == []


def test_missing_fred_key_gives_actionable_failure() -> None:
    r = download_fred_series(SPEC, "2024-01-01", "2024-01-31", FredSettings(), 10, environ={})
    assert r.status == "failed"
    assert "FRED_API_KEY is not set" in r.message and "api_key" in r.message


def test_blank_key_treated_as_missing() -> None:
    assert resolve_fred_api_key(FredSettings(), {"FRED_API_KEY": "  "}) is None


def test_fred_api_key_never_leaks_into_errors() -> None:
    secret = "supersecretkey123"

    def bad_get(url, params, timeout):
        return FakeResponse(status_code=400, text=f"Bad Request. api_key={secret} is invalid")

    r = download_fred_series(SPEC, "2024-01-01", "2024-01-31", FredSettings(max_retries=1), 10,
                             environ={"FRED_API_KEY": secret}, http_get=bad_get, sleep=lambda s: None)
    assert r.status == "failed"
    assert secret not in r.message and secret not in str(r.manifest_entry())


def test_public_csv_fallback_only_when_configured() -> None:
    body = "observation_date,SYNYLD\n2024-01-02,4.10\n2024-01-03,\n2024-01-04,4.15\n"

    def csv_get(url, params, timeout):
        return FakeResponse(text=body)

    off = download_fred_series(SPEC, "2024-01-02", "2024-01-04", FredSettings(), 10, environ={},
                               http_get=csv_get)
    assert off.status == "failed"
    on = download_fred_series(SPEC, "2024-01-02", "2024-01-04",
                              FredSettings(allow_public_csv_fallback=True), 10, environ={}, http_get=csv_get)
    assert on.status == "success" and on.source == "fred_public_csv"
    assert on.frame["value"].tolist()[1] == ""


def test_parse_fred_csv_rejects_wrong_columns() -> None:
    with pytest.raises(Exception, match="Unexpected"):
        parse_fred_csv("a,b\n1,2\n", "SYNYLD")


def test_raw_write_refuses_silent_overwrite_and_archives(tmp_path) -> None:
    path = raw_file_path(tmp_path, "stock", "SYNA")
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "adj_close": [1.0]})
    meta = {"source_id": "SYNA", "requested_start": "2024-01-01", "requested_end": "2024-01-31",
            "status": "success"}
    write_raw_series(frame, path, meta)
    with pytest.raises(FileExistsError):
        write_raw_series(frame, path, meta)
    write_raw_series(frame, path, meta, replace=True)
    archived = list((path.parent / "_archive").iterdir())
    assert len(archived) == 2  # csv + meta
    assert len(read_raw_series(path)) == 1
    assert read_raw_series(path)["date"].iloc[0] == "2024-01-02"


def test_cache_validity_depends_on_request(tmp_path) -> None:
    path = raw_file_path(tmp_path, "macro", "SYNYLD")
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "value": ["."]})
    write_raw_series(frame, path, {"source_id": "SYNYLD", "requested_start": "2024-01-01",
                                   "requested_end": "2024-01-31", "status": "success"})
    assert cache_is_valid(path, "SYNYLD", "2024-01-01", "2024-01-31")
    assert not cache_is_valid(path, "SYNYLD", "2023-01-01", "2024-01-31")
    assert read_raw_series(path)["value"].iloc[0] == "."  # FRED marker preserved raw
