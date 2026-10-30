"""Deterministic stress (Test G, H) and historical replay."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rlstress.config import RiskLimitSpec
from rlstress.exceptions import InsufficientDataError, InvalidShockError, MissingDataError
from rlstress.portfolio import Portfolio
from rlstress.stress import apply_asset_return_shocks, replay_historical_scenario, worst_historical_windows

LIMITS = [
    RiskLimitSpec("loss_5d", "cumulative_loss", "fraction_of_initial_value", 5, 0.10, ">="),
    RiskLimitSpec("dd_5d", "max_drawdown", "fraction_of_peak_value", 5, 0.10, ">="),
    RiskLimitSpec("var95", "historical_var", "fraction_of_portfolio_value", 1, 0.03, ">=", confidence=0.95),
]
P = Portfolio("ab", pd.Series({"A": 0.6, "B": 0.4}), 100_000.0)


def test_g_deterministic_stress() -> None:
    res = apply_asset_return_shocks(P, {"A": -0.10, "B": -0.05}, 5, LIMITS)
    assert res.portfolio_return == pytest.approx(-0.08)
    assert res.loss_usd == pytest.approx(8_000)
    assert res.final_value == pytest.approx(92_000)
    assert res.loss_pct == pytest.approx(0.08)
    contrib = {c["ticker"]: c for c in res.asset_contributions}
    assert contrib["A"]["contribution_usd"] == pytest.approx(-6_000)
    assert contrib["B"]["contribution_usd"] == pytest.approx(-2_000)


def test_h_breach_vs_no_breach() -> None:
    ok = apply_asset_return_shocks(P, {"A": -0.10, "B": -0.05}, 5, LIMITS)
    assert not ok.breached
    bad = apply_asset_return_shocks(P, {"A": -0.15, "B": -0.075}, 5, LIMITS)  # 12% loss
    assert bad.loss_pct == pytest.approx(0.12) and bad.breached
    # 1-day VaR limits are not applied to stress paths at all
    assert {r.limit_name for r in bad.limit_results} == {"loss_5d", "dd_5d"}


def test_h_equality_at_limit() -> None:
    p = Portfolio("noise", pd.Series({"A": 0.3, "B": 0.7}), 100_000.0)
    res = apply_asset_return_shocks(p, {"A": -0.10, "B": -0.10}, 5, LIMITS)
    assert res.loss_pct < 0.10  # 0.09999999999999999 in floating point
    loss = next(r for r in res.limit_results if r.limit_name == "loss_5d")
    assert loss.breached is True  # 10% loss with rule ">=" (float noise absorbed by tolerance)


def test_horizon_mismatch_not_evaluated() -> None:
    res = apply_asset_return_shocks(P, {"A": -0.5, "B": -0.5}, 1, LIMITS)
    assert all(r.status == "not_evaluated" for r in res.limit_results)
    assert not res.breached


def test_gain_scenario() -> None:
    res = apply_asset_return_shocks(P, {"A": 0.10, "B": 0.0}, 5, LIMITS)
    assert res.loss_pct == pytest.approx(-0.06) and res.max_drawdown == 0.0 and not res.breached


@pytest.mark.parametrize("shocks,match", [
    ({"A": -1.2, "B": 0.0}, "-100%"),
    ({"A": -0.1}, "no shock"),
    ({"A": -0.1, "B": 0.0, "C": 0.0}, "not in the portfolio"),
    ({"A": float("nan"), "B": 0.0}, "finite"),
    ({"A": "x", "B": 0.0}, "finite"),
])
def test_invalid_shocks(shocks, match) -> None:
    with pytest.raises(InvalidShockError, match=match):
        apply_asset_return_shocks(P, shocks, 5)


def test_total_loss_shock_allowed() -> None:
    res = apply_asset_return_shocks(P, {"A": -1.0, "B": -1.0}, 5)
    assert res.final_value == pytest.approx(0.0) and res.loss_pct == pytest.approx(1.0)


def test_result_serializes_to_json() -> None:
    res = apply_asset_return_shocks(P, {"A": -0.1, "B": -0.05}, 5, LIMITS)
    payload = json.loads(json.dumps(res.to_dict()))
    assert payload["loss_usd"] == pytest.approx(8000) and payload["breached"] is False
    assert payload["limit_results"][0]["breach_rule"] == ">="


# --- Historical replay ------------------------------------------------------------------------

def _frame() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=8)
    return pd.DataFrame({
        "A": [0.0, 0.01, -0.10, -0.05, 0.08, 0.0, 0.01, 0.02],
        "B": [0.0, 0.00, -0.05, -0.05, 0.02, 0.0, 0.01, np.nan],
    }, index=idx)


def test_replay_values_and_dates() -> None:
    f = _frame()
    res = replay_historical_scenario(P, f, "2024-01-03", 3, LIMITS, name="t")
    daily = [0.6 * -0.10 + 0.4 * -0.05, 0.6 * -0.05 + 0.4 * -0.05, 0.6 * 0.08 + 0.4 * 0.02]
    expected = 100_000 * np.prod([1 + d for d in daily])
    assert res.final_value == pytest.approx(expected)
    assert res.start_date == "2024-01-03" and res.end_date == "2024-01-05"
    assert list(res.value_path)[0] == "2024-01-02"  # base date = previous trading day
    trough = 100_000 * (1 + daily[0]) * (1 + daily[1])
    assert res.max_drawdown == pytest.approx(trough / 100_000 - 1)
    assert res.loss_pct == pytest.approx(1 - expected / 100_000)


def test_replay_starts_on_next_trading_day() -> None:
    res = replay_historical_scenario(P, _frame(), "2024-01-06", 2)  # Saturday -> Monday 2024-01-08
    assert res.start_date == "2024-01-08"


def test_replay_refuses_missing_and_short_windows() -> None:
    with pytest.raises(MissingDataError):
        replay_historical_scenario(P, _frame(), "2024-01-08", 3)
    with pytest.raises(InsufficientDataError):
        replay_historical_scenario(P, _frame(), "2024-01-10", 5)


def test_worst_windows_non_overlapping_and_skip_missing() -> None:
    f = _frame()
    out = worst_historical_windows(f, P.weights, 2, 3)
    assert out.iloc[0]["start_date"] == "2024-01-03" and out.iloc[0]["end_date"] == "2024-01-04"
    expected = (1 + 0.6 * -0.10 + 0.4 * -0.05) * (1 + 0.6 * -0.05 + 0.4 * -0.05) - 1
    assert out.iloc[0]["cumulative_return"] == pytest.approx(expected)
    spans = [(r.start_date, r.end_date) for r in out.itertuples()]
    assert all(e1 < s2 or e2 < s1 for i, (s1, e1) in enumerate(spans) for (s2, e2) in spans[i + 1:])
    assert "2024-01-10" not in set(out["end_date"])  # window with the missing B return skipped
