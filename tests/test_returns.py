"""Test group C/D: returns and macro/VIX transformations against hand calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlstress.data.returns import (
    asset_returns_frame,
    bp_changes,
    log_returns,
    macro_changes_frame,
    safe_pct_changes,
    simple_returns,
    vix_changes_frame,
)


def _series(values, start="2024-01-02", freq="B", name="P"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq), name=name, dtype=float)


def test_simple_returns_hand_example() -> None:
    r = simple_returns(_series([100, 110, 99]))
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(0.10, abs=1e-12)
    assert r.iloc[2] == pytest.approx(-0.10, abs=1e-12)


def test_log_returns_hand_example() -> None:
    r = log_returns(_series([100, 110, 99]))
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(np.log(1.1))
    assert r.iloc[2] == pytest.approx(np.log(0.9))


def test_nonpositive_prices_raise_by_default() -> None:
    with pytest.raises(ValueError, match="non-positive"):
        simple_returns(_series([100, 0, 99]))
    with pytest.raises(ValueError):
        simple_returns(_series([100, -5, 99]))


def test_nonpositive_prices_nan_policy() -> None:
    r = simple_returns(_series([20, -37, 9, 10]), nonpositive="nan")
    assert r.iloc[1:3].isna().all()  # into and out of the negative price
    assert r.iloc[3] == pytest.approx(10 / 9 - 1)


def test_missing_price_never_becomes_zero_return() -> None:
    r = simple_returns(_series([100, np.nan, 105, 110]))
    assert r.iloc[1:3].isna().all()  # no return on or right after the gap
    assert r.iloc[3] == pytest.approx(110 / 105 - 1)
    assert not (r == 0).any()


def test_returns_after_long_calendar_gap_are_undefined() -> None:
    prices = pd.Series([100.0, 101.0, 102.0],
                       index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-15"]))
    r = simple_returns(prices, max_gap_days=5)
    assert r.iloc[1] == pytest.approx(0.01)
    assert np.isnan(r.iloc[2])


def test_weekend_gap_within_limit_is_allowed() -> None:
    prices = pd.Series([100.0, 102.0], index=pd.to_datetime(["2024-01-05", "2024-01-08"]))
    assert simple_returns(prices, max_gap_days=5).iloc[1] == pytest.approx(0.02)


def test_asset_returns_frame_column_names() -> None:
    frame = pd.DataFrame({"AAPL_adj_close": [100.0, 110.0], "MSFT_adj_close": [50.0, 55.0]},
                         index=pd.date_range("2024-01-02", periods=2))
    out = asset_returns_frame(frame, include_log=True, max_gap_days=5)
    assert list(out.columns) == ["AAPL_return", "AAPL_log_return", "MSFT_return", "MSFT_log_return"]


def test_yield_change_in_basis_points() -> None:
    change = bp_changes(_series([4.10, 4.15, 4.05]))
    assert np.isnan(change.iloc[0])
    assert change.iloc[1] == pytest.approx(5.0)
    assert change.iloc[2] == pytest.approx(-10.0)


def test_macro_changes_frame_rate_and_price() -> None:
    rate = macro_changes_frame(_series([5.33, 5.33, 5.08]), "DFF", "rate", True, 5)
    assert list(rate.columns) == ["DFF_change_bp"]
    assert rate["DFF_change_bp"].iloc[1] == pytest.approx(0.0)  # unchanged rate is a valid 0 bp
    assert rate["DFF_change_bp"].iloc[2] == pytest.approx(-25.0)
    wti = macro_changes_frame(_series([80.0, 88.0]), "WTI", "price", True, 5)
    assert wti["WTI_return"].iloc[1] == pytest.approx(0.10)
    assert wti["WTI_log_return"].iloc[1] == pytest.approx(np.log(1.1))


def test_vix_changes_handle_invalid_denominator() -> None:
    out = vix_changes_frame(_series([20.0, 0.0, 15.0, 18.0]), "VIX", 5)
    assert out["VIX_change"].tolist()[1:] == pytest.approx([-20.0, 15.0, 3.0])
    pct = out["VIX_pct_change"]
    assert pct.iloc[1] == pytest.approx(-1.0)
    assert np.isnan(pct.iloc[2])  # previous level 0 -> undefined, not inf
    assert pct.iloc[3] == pytest.approx(0.2)
    assert np.isfinite(pct.dropna()).all()


def test_safe_pct_changes_negative_denominator() -> None:
    assert np.isnan(safe_pct_changes(_series([-1.0, 2.0])).iloc[1])
