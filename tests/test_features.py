"""Test group E: rolling features are causal, respect min_periods and are named consistently."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlstress import features as feat


def _series(values, name="x"):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), name=name, dtype=float)


def test_rolling_volatility_matches_manual_and_annualises() -> None:
    r = _series([0.01, -0.02, 0.03, 0.00])
    vol = feat.rolling_volatility(r, window=3, min_periods=3, annualization_days=252)
    expected = np.std([-0.02, 0.03, 0.00], ddof=1) * np.sqrt(252)
    assert vol.iloc[3] == pytest.approx(expected)
    assert vol.iloc[:2].isna().all()


def test_insufficient_observations_give_undefined() -> None:
    r = _series([0.01, np.nan, np.nan, 0.02, 0.03])
    m = feat.rolling_mean(r, window=3, min_periods=2)
    assert np.isnan(m.iloc[2])  # only one valid value in window
    assert m.iloc[4] == pytest.approx(0.025)


def test_rolling_features_ignore_future_values() -> None:
    r = _series(np.random.default_rng(42).normal(0, 0.01, 50))
    full = feat.rolling_volatility(r, 10, 8)
    changed = r.copy()
    changed.iloc[30:] = 5.0  # wildly different future
    trunc = feat.rolling_volatility(changed, 10, 8)
    pd.testing.assert_series_equal(full.iloc[:30], trunc.iloc[:30])


def test_correlation_uses_aligned_observations_only() -> None:
    x = _series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    y = _series([2.0, 4.0, 6.0, 8.0, 10.0, np.nan])
    c = feat.rolling_correlation(x, y, window=6, min_periods=5)
    # the x outlier on the date y is missing must not enter the correlation
    assert c.iloc[5] == pytest.approx(1.0)


def test_correlation_undefined_with_too_few_pairs() -> None:
    x = _series([1.0, 2.0, 3.0, 4.0])
    y = _series([1.0, np.nan, np.nan, 2.0])
    assert feat.rolling_correlation(x, y, 4, 3).isna().all()


def test_drawdown_from_trailing_high() -> None:
    p = _series([100.0, 120.0, 90.0, 60.0])
    dd = feat.rolling_drawdown(p, window=2, min_periods=2)
    assert dd.iloc[2] == pytest.approx(90 / 120 - 1)
    assert dd.iloc[3] == pytest.approx(60 / 90 - 1)  # 120 has left the 2-day window


def test_feature_column_names() -> None:
    returns = pd.DataFrame({"AAA_return": [0.01, 0.02, 0.0], "SPY_return": [0.01, 0.0, 0.01]},
                           index=pd.bdate_range("2024-01-01", periods=3))
    mp = {2: 2}
    rf = feat.return_features(returns, [2], mp, 252)
    assert set(rf.columns) == {"AAA_vol_ann_2d", "AAA_ret_mean_2d", "SPY_vol_ann_2d", "SPY_ret_mean_2d"}
    corr, skipped = feat.correlation_features(returns, {"AAA": ["SPY", "XLI"]}, [2], mp)
    assert list(corr.columns) == ["AAA_SPY_corr_2d"]
    assert skipped == ["AAA-XLI"]


def test_regime_thresholds_and_labels() -> None:
    train = _series([10.0, 12.0, 14.0, 16.0, 18.0])
    t = feat.fit_quantile_thresholds(train, [0.25, 0.75])
    assert t == pytest.approx([12.0, 16.0])
    labels = feat.assign_regimes(_series([11.0, 12.0, 15.0, 30.0, np.nan]), t, ["low", "normal", "high"])
    assert labels.tolist()[:4] == ["low", "low", "normal", "high"]
    assert pd.isna(labels.iloc[4])
