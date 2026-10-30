"""Portfolio validation, return aggregation and valuation (Tests A, B, C, J, K)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlstress.exceptions import MissingDataError, PortfolioValidationError, ReturnDataError
from rlstress.portfolio import (
    Portfolio,
    apply_missing_data_policy,
    base_date_for,
    buy_and_hold_value,
    calculate_asset_returns,
    calculate_portfolio_returns,
    calculate_portfolio_value,
    normalize_weights,
    select_window,
    validate_confidence,
    validate_horizon,
    validate_weights,
    value_consistency_error,
)

EIGHT = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "WMT", "CAT"]


def _dates(n: int, start: str = "2024-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _two_asset(returns_a, returns_b) -> pd.DataFrame:
    return pd.DataFrame({"A": returns_a, "B": returns_b}, index=_dates(len(returns_a)))


# --- Test A / B: aggregation and value ------------------------------------------------------

def test_a_portfolio_return_aggregation() -> None:
    r = calculate_portfolio_returns(_two_asset([0.10], [-0.05]), pd.Series({"A": 0.6, "B": 0.4}))
    assert r.iloc[0] == pytest.approx(0.6 * 0.10 + 0.4 * -0.05)  # = 0.04
    assert r.iloc[0] == pytest.approx(0.04)


def test_b_portfolio_value() -> None:
    r = pd.Series([0.04], index=_dates(1))
    v = calculate_portfolio_value(r, 100_000, "2024-01-01")
    assert v.iloc[0] == 100_000 and v.iloc[-1] == pytest.approx(104_000)
    assert v.index[0] == pd.Timestamp("2024-01-01")


def test_c_equal_weight_eight_assets() -> None:
    weights = validate_weights({t: 0.125 for t in EIGHT}, universe=EIGHT)
    assert weights.sum() == pytest.approx(1.0)
    rets = [0.01, -0.02, 0.03, 0.0, -0.01, 0.02, 0.005, -0.015]
    frame = pd.DataFrame([rets], columns=EIGHT, index=_dates(1))
    r = calculate_portfolio_returns(frame, weights)
    assert r.iloc[0] == pytest.approx(np.mean(rets))


def test_column_order_does_not_matter() -> None:
    frame = _two_asset([0.10], [-0.05])[["B", "A"]]
    r = calculate_portfolio_returns(frame, pd.Series({"A": 0.6, "B": 0.4}))
    assert r.iloc[0] == pytest.approx(0.04)


def test_value_path_consistent_with_returns() -> None:
    r = pd.Series([0.01, -0.02, 0.03], index=_dates(3))
    v = calculate_portfolio_value(r, 1000.0, "2024-01-01")
    assert value_consistency_error(v, r) < 1e-15
    assert v.iloc[-1] == pytest.approx(1000 * 1.01 * 0.98 * 1.03)


def test_constant_mix_differs_from_buy_and_hold() -> None:
    frame = _two_asset([1.0, -0.5], [0.0, 0.0])
    w = pd.Series({"A": 0.5, "B": 0.5})
    cm = calculate_portfolio_value(calculate_portfolio_returns(frame, w), 100.0, "2024-01-01")
    bh = buy_and_hold_value(frame, w, 100.0, "2024-01-01")
    # constant mix: +50% then -25% -> 112.5 ; buy and hold: A 50->100->50, B 50 -> 100
    assert cm.iloc[-1] == pytest.approx(112.5)
    assert bh["total"].iloc[-1] == pytest.approx(100.0)


def test_base_date_must_precede_returns() -> None:
    with pytest.raises(ReturnDataError):
        calculate_portfolio_value(pd.Series([0.01], index=_dates(1)), 100.0, "2024-01-02")


def test_base_date_for_uses_previous_calendar_day() -> None:
    cal = _dates(5)
    assert base_date_for(cal, cal[2]) == cal[1]
    with pytest.raises(ReturnDataError):
        base_date_for(cal, cal[0])


# --- Validation (Test K) ---------------------------------------------------------------------

@pytest.mark.parametrize("weights,match", [
    ({"A": 0.6, "B": 0.5}, "sum"),
    ({"A": float("nan"), "B": 0.5}, "finite"),
    ({"A": float("inf"), "B": 0.5}, "finite"),
    ({"A": 1.2, "B": -0.2}, "negative"),
    ({"A": "0.5", "B": 0.5}, "number"),
    ({"A": True, "B": 0.0}, "number"),
    ({}, "no holdings"),
])
def test_invalid_weights_rejected(weights, match) -> None:
    with pytest.raises(PortfolioValidationError, match=match):
        validate_weights(weights)


def test_tolerance_is_documented_and_applied() -> None:
    validate_weights({"A": 0.5, "B": 0.5 + 5e-9}, tolerance=1e-8)
    with pytest.raises(PortfolioValidationError):
        validate_weights({"A": 0.5, "B": 0.5 + 5e-8}, tolerance=1e-8)


def test_unknown_ticker_and_unsupported_policy() -> None:
    with pytest.raises(PortfolioValidationError, match="universe"):
        validate_weights({"ZZZ": 1.0}, universe=EIGHT)
    with pytest.raises(PortfolioValidationError, match="not implemented"):
        validate_weights({"A": 1.5, "B": -0.5}, policy="long_short")


def test_no_silent_normalisation_but_explicit_operation_available() -> None:
    with pytest.raises(PortfolioValidationError):
        Portfolio("p", pd.Series({"A": 2.0, "B": 2.0}), 100.0)
    normalised = normalize_weights({"A": 2.0, "B": 2.0})
    assert normalised.tolist() == [0.5, 0.5]
    assert Portfolio("p", normalised, 100.0).weights.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("capital", [0, -5, float("nan"), float("inf")])
def test_invalid_capital(capital) -> None:
    with pytest.raises(PortfolioValidationError, match="capital"):
        Portfolio("p", pd.Series({"A": 1.0}), capital)


@pytest.mark.parametrize("alpha", [0, 1, 1.5, -0.1])
def test_invalid_confidence(alpha) -> None:
    with pytest.raises(PortfolioValidationError):
        validate_confidence(alpha)


@pytest.mark.parametrize("h", [0, -1, 2.5, True])
def test_invalid_horizon(h) -> None:
    with pytest.raises(PortfolioValidationError):
        validate_horizon(h)


def test_return_frame_validation() -> None:
    good = pd.DataFrame({"A_return": [0.01, 0.02]}, index=_dates(2))
    calculate_asset_returns(good, ["A"])
    dup = pd.DataFrame({"A_return": [0.01, 0.02]}, index=pd.DatetimeIndex(["2024-01-02"] * 2))
    with pytest.raises(ReturnDataError, match="duplicate"):
        calculate_asset_returns(dup, ["A"])
    with pytest.raises(ReturnDataError, match="sorted"):
        calculate_asset_returns(good.iloc[::-1], ["A"])
    with pytest.raises(ReturnDataError, match="lack columns"):
        calculate_asset_returns(good, ["A", "B"])
    with pytest.raises(ReturnDataError, match="empty"):
        calculate_asset_returns(good.iloc[:0], ["A"])
    with pytest.raises(ReturnDataError, match="-100%"):
        calculate_asset_returns(pd.DataFrame({"A_return": [-1.5]}, index=_dates(1)), ["A"])
    with pytest.raises(ReturnDataError, match="DatetimeIndex"):
        calculate_asset_returns(pd.DataFrame({"A_return": [0.1]}), ["A"])


def test_other_columns_do_not_trigger_return_checks() -> None:
    frame = pd.DataFrame({"A_return": [0.01], "DGS10_change_bp": [-25.0]}, index=_dates(1))
    assert calculate_asset_returns(frame, ["A"]).columns.tolist() == ["A"]


def test_misaligned_holdings_rejected() -> None:
    with pytest.raises(ReturnDataError, match="do not match"):
        calculate_portfolio_returns(_two_asset([0.1], [0.1]), pd.Series({"A": 0.5, "C": 0.5}))


# --- Missing data (Test J) -------------------------------------------------------------------

def test_missing_returns_never_zero_filled() -> None:
    frame = _two_asset([0.01, np.nan, 0.03], [0.02, 0.02, 0.01])
    w = pd.Series({"A": 0.5, "B": 0.5})
    with pytest.raises(MissingDataError):
        calculate_portfolio_returns(frame, w)
    aligned = apply_missing_data_policy(frame, "complete_case")
    r = calculate_portfolio_returns(aligned.returns, w)
    assert len(r) == 2 and frame.index[1] not in r.index
    assert aligned.excluded_dates == [frame.index[1].date().isoformat()]
    assert r.loc[frame.index[2]] == pytest.approx(0.02)  # not affected by the gap


def test_raise_policy() -> None:
    frame = _two_asset([0.01, np.nan], [0.02, 0.02])
    with pytest.raises(MissingDataError):
        apply_missing_data_policy(frame, "raise")
    with pytest.raises(ReturnDataError):
        apply_missing_data_policy(frame, "zero_fill")


def test_select_window_inclusive() -> None:
    frame = _two_asset([0.1] * 5, [0.1] * 5)
    out = select_window(frame, frame.index[1].isoformat(), frame.index[3].isoformat())
    assert list(out.index) == list(frame.index[1:4])
