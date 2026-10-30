"""Volatility, historical VaR/CVaR and drawdown against hand-derived values (Tests D, E, F)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlstress.exceptions import InsufficientDataError, MissingDataError
from rlstress.risk import (
    calculate_historical_cvar,
    calculate_historical_var,
    calculate_historical_volatility,
    calculate_max_drawdown,
    rolling_volatility_table,
)
from rlstress.risk.var import var_order_index


def _series(values) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


# Ten returns, deliberately unsorted. Sorted: -5, -3, -2, -1, 0, 1, 2, 3, 4, 5 (%).
TEN = _series([0.02, -0.03, 0.05, -0.01, 0.0, -0.05, 0.04, 0.01, -0.02, 0.03])


# --- Volatility ----------------------------------------------------------------------------

def test_volatility_hand_example() -> None:
    r = _series([0.01, -0.02, 0.03, 0.0])
    res = calculate_historical_volatility(r, 252)
    mean = 0.005
    manual = np.sqrt(sum((x - mean) ** 2 for x in [0.01, -0.02, 0.03, 0.0]) / 3)  # ddof = 1
    assert res.daily_std == pytest.approx(manual)
    assert res.annualized == pytest.approx(manual * np.sqrt(252))
    assert res.n_observations == 4


def test_volatility_input_checks() -> None:
    with pytest.raises(InsufficientDataError):
        calculate_historical_volatility(_series([0.01]))
    with pytest.raises(MissingDataError):
        calculate_historical_volatility(_series([0.01, np.nan, 0.02]))


def test_rolling_volatility_requires_full_window() -> None:
    table = rolling_volatility_table(_series([0.01, -0.02, 0.03, 0.0, 0.01]), [3], 252)
    assert list(table.columns) == ["portfolio_vol_3d", "portfolio_vol_ann_3d"]
    assert table["portfolio_vol_3d"].iloc[:2].isna().all()
    assert table["portfolio_vol_3d"].iloc[2] == pytest.approx(np.std([0.01, -0.02, 0.03], ddof=1))


# --- VaR (Test D) --------------------------------------------------------------------------

@pytest.mark.parametrize("alpha,k,expected", [
    (0.95, 1, 0.05),  # n(1-a) = 0.5 -> k = ceil(0.5) = 1 -> -r_(1) = 5%
    (0.80, 2, 0.03),  # n(1-a) = 2   -> k = 2 -> -r_(2) = 3%
    (0.75, 3, 0.02),  # n(1-a) = 2.5 -> k = 3 -> -r_(3) = 2%
])
def test_var_hand_examples(alpha, k, expected) -> None:
    res = calculate_historical_var(TEN, alpha, capital=100_000)
    assert res.order_index == k
    assert res.var_pct == pytest.approx(expected)
    assert res.var_usd == pytest.approx(expected * 100_000)
    assert res.n_observations == 10


def test_var_is_kth_order_statistic_at_integer_tail() -> None:
    r = _series(np.random.default_rng(42).normal(0, 0.01, 1000))
    srt = np.sort(r.to_numpy())
    assert calculate_historical_var(r, 0.95, 1.0).var_pct == pytest.approx(-srt[49])  # k = 50
    assert calculate_historical_var(r, 0.99, 1.0).var_pct == pytest.approx(-srt[9])   # k = 10
    # numpy with p = 1 - 0.95 = 0.050000000000000044 picks k = 51 here: a float artifact
    assert -np.quantile(srt, 1 - 0.95, method="inverted_cdf") == pytest.approx(-srt[50])


def test_var_matches_numpy_away_from_integer_boundaries() -> None:
    r = _series(np.random.default_rng(7).normal(0, 0.01, 1003))
    for a in (0.95, 0.99):
        assert calculate_historical_var(r, a, 1.0).var_pct == pytest.approx(
            -np.quantile(r.to_numpy(), 1 - a, method="inverted_cdf"))


def test_order_index_guards_float_noise() -> None:
    assert 100 * (1 - 0.95) > 5  # binary floating point gives 5.000000000000004
    assert var_order_index(100, 0.95) == 5


def test_negative_var_is_preserved() -> None:
    res = calculate_historical_var(_series([0.01, 0.02, 0.03, 0.04, 0.05]), 0.8, 1000.0)
    assert res.var_pct == pytest.approx(-0.01)  # k = 1: worst day was a 1% gain
    assert any("negative VaR" in w for w in res.warnings)


def test_var_input_checks() -> None:
    with pytest.raises(InsufficientDataError):
        calculate_historical_var(TEN, 0.95, 1.0, min_observations=11)
    with pytest.raises(InsufficientDataError):
        calculate_historical_var(_series([]), 0.95, 1.0)
    with pytest.raises(MissingDataError):
        calculate_historical_var(_series([0.01, np.nan]), 0.95, 1.0)
    assert calculate_historical_var(TEN, 0.95, 1.0, min_tail_observations=10).warnings


# --- CVaR (Test E) -------------------------------------------------------------------------

@pytest.mark.parametrize("alpha,expected,j,boundary", [
    (0.80, (0.05 + 0.03) / 2, 2, 0.0),             # m = 2: average of the two worst losses
    (0.75, (0.05 + 0.03 + 0.5 * 0.02) / 2.5, 2, 0.5),  # m = 2.5: half weight on the 3rd worst
    (0.95, 0.05, 0, 0.5),                          # m = 0.5: half of the worst loss / 0.5
])
def test_cvar_hand_examples(alpha, expected, j, boundary) -> None:
    res = calculate_historical_cvar(TEN, alpha, capital=100_000)
    assert res.cvar_pct == pytest.approx(expected)
    assert res.cvar_usd == pytest.approx(expected * 100_000)
    assert res.n_full_tail_observations == j
    assert res.boundary_weight == pytest.approx(boundary)
    assert res.var_pct == pytest.approx(calculate_historical_var(TEN, alpha, 1.0).var_pct)
    assert res.cvar_pct >= res.var_pct


def test_cvar_ties_at_boundary() -> None:
    # sorted: -4, -4, -4, -1, 2 (%); alpha 0.5 -> m = 2.5, j = 2, boundary weight 0.5 on a tied -4%
    r = _series([-0.04, 0.02, -0.04, -0.01, -0.04])
    res = calculate_historical_cvar(r, 0.5, 1.0)
    assert res.cvar_pct == pytest.approx((0.04 + 0.04 + 0.5 * 0.04) / 2.5)
    assert res.cvar_pct == pytest.approx(0.04)
    assert res.var_pct == pytest.approx(0.04)


def test_cvar_at_least_var_on_random_samples() -> None:
    rng = np.random.default_rng(42)
    for n in (7, 50, 333):
        r = _series(rng.standard_t(3, n) * 0.01)
        for a in (0.9, 0.95, 0.99):
            res = calculate_historical_cvar(r, a, 1.0)
            assert res.cvar_pct >= res.var_pct - 1e-15


def test_cvar_sign_convention_with_gains_only() -> None:
    res = calculate_historical_cvar(_series([0.01, 0.02, 0.03, 0.04, 0.05]), 0.6, 1.0)
    assert res.cvar_pct == pytest.approx(-(0.01 + 0.02) / 2)  # negative: tail average is a gain


def test_cvar_small_sample_warns() -> None:
    res = calculate_historical_cvar(TEN, 0.95, 1.0)
    assert any("below one observation" in w for w in res.warnings)


# --- Drawdown (Test F) ---------------------------------------------------------------------

def test_f_max_drawdown_no_recovery() -> None:
    v = _series([100, 120, 90, 105, 80])
    res = calculate_max_drawdown(v)
    assert res.max_drawdown == pytest.approx(80 / 120 - 1)  # -33.33%
    assert res.peak_value == 120 and res.trough_value == 80
    assert res.peak_date == v.index[1].date().isoformat()
    assert res.trough_date == v.index[4].date().isoformat()
    assert res.recovery_date is None and res.recovered is False


def test_drawdown_with_recovery() -> None:
    v = _series([100, 120, 90, 119, 125])
    res = calculate_max_drawdown(v)
    assert res.max_drawdown == pytest.approx(-0.25)
    assert res.recovery_date == v.index[4].date().isoformat() and res.recovered


def test_drawdown_uses_most_recent_equal_peak() -> None:
    v = _series([100, 120, 110, 120, 90])
    assert calculate_max_drawdown(v).peak_date == v.index[3].date().isoformat()


def test_no_drawdown() -> None:
    res = calculate_max_drawdown(_series([100, 101, 102]))
    assert res.max_drawdown == 0.0 and res.peak_date is None and res.recovered is None


def test_drawdown_input_checks() -> None:
    with pytest.raises(InsufficientDataError):
        calculate_max_drawdown(_series([100]))
    with pytest.raises(MissingDataError):
        calculate_max_drawdown(_series([100, np.nan, 90]))
