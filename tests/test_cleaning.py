"""Test group B: date handling, duplicates, invalid values and missing data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlstress.config import CleaningSettings, MacroSeriesSpec
from rlstress.data.cleaning import clean_fred_frame, clean_yahoo_frame
from rlstress.data.sources import DataValidationError
from rlstress.data.validation import missingness, unexpected_gaps

SETTINGS = CleaningSettings()


def _yahoo(rows):
    cols = ["date", "open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits"]
    return pd.DataFrame(rows, columns=cols).astype({"date": str})


def _row(date, px=100.0, vol=1000):
    return [date, px, px * 1.01, px * 0.99, px, px * 0.98, vol, 0.0, 0.0]


def test_dates_parsed_and_sorted() -> None:
    frame, log = clean_yahoo_frame(_yahoo([_row("2024-01-03"), _row("2024-01-02")]), "T", "stock", SETTINGS)
    assert list(frame.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert frame.index.tz is None
    assert (frame.index == frame.index.normalize()).all()


def test_invalid_dates_dropped_and_counted() -> None:
    frame, log = clean_yahoo_frame(_yahoo([_row("2024-01-02"), _row("not-a-date")]), "T", "stock", SETTINGS)
    assert len(frame) == 1
    assert log.invalid_dates_dropped == 1


def test_exact_duplicates_removed() -> None:
    frame, log = clean_yahoo_frame(_yahoo([_row("2024-01-02"), _row("2024-01-02"), _row("2024-01-03")]),
                                   "T", "stock", SETTINGS)
    assert len(frame) == 2
    assert log.exact_duplicates_removed == 1
    assert log.conflicting_duplicate_dates == []


def test_conflicting_duplicates_dropped_and_reported() -> None:
    raw = _yahoo([_row("2024-01-02", 100.0), _row("2024-01-02", 101.0), _row("2024-01-03")])
    frame, log = clean_yahoo_frame(raw, "T", "stock", SETTINGS)
    assert pd.Timestamp("2024-01-02") not in frame.index
    assert log.conflicting_duplicate_dates == ["2024-01-02"]


def test_conflicting_duplicates_can_raise() -> None:
    raw = _yahoo([_row("2024-01-02", 100.0), _row("2024-01-02", 101.0)])
    with pytest.raises(DataValidationError, match="conflicting"):
        clean_yahoo_frame(raw, "T", "stock", CleaningSettings(conflict_policy="raise"))


def test_invalid_prices_and_volume_set_missing() -> None:
    rows = [_row("2024-01-02"), _row("2024-01-03", px=-1.0), _row("2024-01-04", vol=-5)]
    frame, log = clean_yahoo_frame(_yahoo(rows), "T", "stock", SETTINGS)
    assert np.isnan(frame.loc["2024-01-03", "adj_close"])
    assert np.isnan(frame.loc["2024-01-04", "volume"])
    assert log.invalid_set_missing["adj_close_nonpositive"] == 1
    assert log.invalid_set_missing["volume_negative"] == 1
    assert len(frame) == 3  # rows kept so gaps stay visible


def test_non_numeric_and_infinite_values_counted() -> None:
    raw = _yahoo([_row("2024-01-02"), _row("2024-01-03")])
    raw["close"] = raw["close"].astype(object)
    raw.loc[0, "close"] = "abc"
    raw.loc[1, "adj_close"] = np.inf
    frame, log = clean_yahoo_frame(raw, "T", "stock", SETTINGS)
    assert log.conversion_failures == {"close": 1}
    assert log.nonfinite_values == {"adj_close": 1}
    assert frame[["close", "adj_close"]].isna().sum().sum() == 2


def test_yahoo_holiday_placeholder_rows_dropped_and_counted() -> None:
    placeholder = ["2024-01-15", np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0.0, 0.0]
    raw = _yahoo([_row("2024-01-12"), placeholder, _row("2024-01-16")])
    frame, log = clean_yahoo_frame(raw, "VIX", "volatility", SETTINGS)
    assert pd.Timestamp("2024-01-15") not in frame.index
    assert log.no_observation_rows_dropped == 1
    assert log.flags["adj_close_missing"] == 0


def test_invalid_price_row_not_treated_as_placeholder() -> None:
    raw = _yahoo([_row("2024-01-12"), _row("2024-01-16", px=-1.0)])
    frame, log = clean_yahoo_frame(raw, "T", "stock", SETTINGS)
    assert pd.Timestamp("2024-01-16") in frame.index  # invalid, kept as visible gap
    assert log.no_observation_rows_dropped == 0


def test_missing_prices_not_filled() -> None:
    raw = _yahoo([_row("2024-01-02"), _row("2024-01-03"), _row("2024-01-04")])
    raw.loc[1, "adj_close"] = np.nan
    frame, log = clean_yahoo_frame(raw, "T", "stock", SETTINGS)
    assert np.isnan(frame.loc["2024-01-03", "adj_close"])
    assert log.flags["adj_close_missing"] == 1


def test_ohlc_inconsistency_flagged_not_changed() -> None:
    row = _row("2024-01-02")
    row[2] = 50.0  # high below low/close
    frame, log = clean_yahoo_frame(_yahoo([row]), "T", "stock", SETTINGS)
    assert log.flags["ohlc_inconsistent"] == 1
    assert frame.iloc[0]["high"] == 50.0


def test_fred_markers_dropped_and_negative_price_kept() -> None:
    spec = MacroSeriesSpec("DCOILWTICO", "WTI", "WTI", "price", "usd_per_barrel")
    raw = pd.DataFrame({"date": ["2020-04-17", "2020-04-20", "2020-04-21", "2020-04-22"],
                        "value": ["18.31", "-36.98", ".", "13.64"]})
    frame, log = clean_fred_frame(raw, spec, SETTINGS)
    assert list(frame.columns) == ["WTI_price_usd_per_barrel"]
    assert frame.loc["2020-04-20", "WTI_price_usd_per_barrel"] == -36.98
    assert pd.Timestamp("2020-04-21") not in frame.index
    assert log.source_missing_markers == 1
    assert log.no_observation_rows_dropped == 1
    assert log.flags["nonpositive_price"] == 1


def test_fred_suspicious_yield_flagged() -> None:
    spec = MacroSeriesSpec("DGS10", "DGS10", "10Y", "yield", "percent")
    raw = pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "value": ["4.10", "40.0"]})
    frame, log = clean_fred_frame(raw, spec, SETTINGS)
    assert log.flags["outside_plausible_range"] == 1
    assert len(frame) == 2


def test_fred_garbage_counted_as_conversion_failure() -> None:
    spec = MacroSeriesSpec("DGS10", "DGS10", "10Y", "yield", "percent")
    raw = pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "value": ["4.10", "n/a"]})
    _, log = clean_fred_frame(raw, spec, SETTINGS)
    assert log.conversion_failures == {"value": 1}
    assert log.source_missing_markers == 0


def test_missingness_and_unexpected_gaps() -> None:
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan]})
    m = missingness(df)
    assert m.loc["a", "n_missing"] == 2
    assert m.loc["a", "pct_missing"] == 50.0
    calendar = pd.bdate_range("2024-01-01", "2024-01-12")
    source = calendar.delete(3)
    gaps = unexpected_gaps(source, calendar)
    assert list(gaps) == [calendar[3]]  # weekends are not gaps
