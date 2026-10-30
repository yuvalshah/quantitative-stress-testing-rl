"""Test groups F and G: date alignment and chronological, leakage-aware splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlstress.config import SplitSettings
from rlstress.data.alignment import (
    align_to_calendar,
    complete_cases,
    limited_forward_fill,
    market_calendar,
    wide_frame,
)
from rlstress.data.splits import (
    apply_standardizer,
    assign_split_labels,
    check_split_integrity,
    fit_standardizer,
    split_frame,
    split_summary,
)

SPLITS = SplitSettings(train=("2024-01-01", "2024-01-10"),
                       validation=("2024-01-11", "2024-01-17"),
                       test=("2024-01-18", "2024-01-31"))


def _frame(dates, col, values=None):
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="date")
    vals = values if values is not None else np.arange(len(idx), dtype=float) + 1
    return pd.DataFrame({col: vals}, index=idx)


def test_outer_join_preserves_source_observations() -> None:
    a = _frame(["2024-01-02", "2024-01-03"], "adj_close")
    b = _frame(["2024-01-03", "2024-01-04"], "adj_close")
    wide = wide_frame({"A": a, "B": b}, {"adj_close": "{alias}_adj_close"})
    assert list(wide.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    assert wide["A_adj_close"].notna().sum() == 2
    assert wide["B_adj_close"].notna().sum() == 2


def test_calendar_creates_no_unexplained_dates() -> None:
    a = _frame(["2024-01-02", "2024-01-04"], "x")
    b = _frame(["2024-01-03"], "x")
    cal = market_calendar([a, b])
    assert list(cal) == list(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))


def test_align_reports_excluded_dates_and_keeps_source_intact() -> None:
    cal = pd.DatetimeIndex(pd.to_datetime(["2024-01-05", "2024-01-08"]))
    macro = _frame(["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"], "DFF_rate_pct")
    before = macro.copy()
    aligned, excluded = align_to_calendar(macro, cal)
    assert list(excluded) == list(pd.to_datetime(["2024-01-06", "2024-01-07"]))
    assert len(aligned) == 2
    pd.testing.assert_frame_equal(macro, before)


def test_alignment_does_not_fill() -> None:
    cal = pd.DatetimeIndex(pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10"]))
    s = _frame(["2024-01-08", "2024-01-10"], "DGS10_yield_pct", [4.0, 4.2])
    aligned, _ = align_to_calendar(s, cal)
    assert np.isnan(aligned.loc["2024-01-09", "DGS10_yield_pct"])
    assert aligned["DGS10_yield_pct"].isna().sum() == 1


def test_limited_forward_fill_flags() -> None:
    df = pd.DataFrame({"y": [1.0, np.nan, np.nan, 2.0]}, index=pd.bdate_range("2024-01-01", periods=4))
    assert limited_forward_fill(df, ["y"], 0) is df
    filled = limited_forward_fill(df, ["y"], 1)
    assert filled["y"].tolist()[:2] == [1.0, 1.0] and np.isnan(filled["y"].iloc[2])
    assert filled["y_is_filled"].tolist() == [False, True, False, False]


def test_complete_cases_count() -> None:
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, np.nan]})
    assert len(complete_cases(df)) == 1
    assert len(complete_cases(df, ["a"])) == 2


def _daily(start="2024-01-01", end="2024-01-31"):
    idx = pd.bdate_range(start, end, name="date")
    return pd.DataFrame({"x": np.arange(len(idx), dtype=float)}, index=idx)


def test_splits_chronological_disjoint_and_complete() -> None:
    df = _daily()
    parts = split_frame(df, SPLITS)
    check_split_integrity(parts)
    assert sum(len(p) for p in parts.values()) == len(df)
    assert parts["train"].index.max() < parts["validation"].index.min()
    assert parts["validation"].index.max() < parts["test"].index.min()
    assert split_summary(df, SPLITS)["unassigned"]["n_rows"] == 0


def test_dates_outside_splits_are_reported() -> None:
    df = _daily("2024-01-01", "2024-02-05")
    summary = split_summary(df, SPLITS)
    assert summary["unassigned"]["n_rows"] == 3
    assert summary["unassigned"]["dates"][0] == "2024-02-01"


def test_integrity_check_detects_overlap() -> None:
    df = _daily()
    with pytest.raises(ValueError):
        check_split_integrity({"train": df.iloc[:10], "validation": df.iloc[5:15]})


def test_overlapping_split_config_rejected() -> None:
    bad = SplitSettings(train=("2024-01-01", "2024-01-15"), validation=("2024-01-10", "2024-01-20"),
                        test=("2024-01-21", "2024-01-31"))
    with pytest.raises(ValueError, match="overlaps"):
        assign_split_labels(_daily().index, bad)


def test_standardizer_fitted_on_training_only() -> None:
    df = _daily()
    df.loc[df.index >= "2024-01-11", "x"] = 1000.0  # extreme later values
    parts = split_frame(df, SPLITS)
    params = fit_standardizer(parts["train"], ["x"])
    assert params["x"]["mean"] == pytest.approx(parts["train"]["x"].mean())
    scaled_test = apply_standardizer(parts["test"], params)
    expected = (1000.0 - params["x"]["mean"]) / params["x"]["std"]
    assert scaled_test["x"].iloc[0] == pytest.approx(expected)
