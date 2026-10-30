"""Asset contributions (Test I) and risk-limit evaluation (Test H boundary rules)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlstress.config import RiskLimitSpec
from rlstress.exceptions import IncompatibleMetricError, MissingDataError
from rlstress.portfolio import calculate_portfolio_returns
from rlstress.risk import (
    MetricObservation,
    calculate_asset_risk_contributions,
    contribution_reconciliation_error,
    evaluate_limit,
    evaluate_risk_limits,
    extreme_loss_contributions,
    is_breach,
    return_contributions,
    volatility_contributions,
)

LOSS_LIMIT = RiskLimitSpec("loss_5d", "cumulative_loss", "fraction_of_initial_value", 5, 0.10, ">=")


def _returns(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2020-01-01", periods=n)
    base = rng.normal(0, 0.01, n)
    return pd.DataFrame({
        "LOWVOL": base * 0.5 + rng.normal(0, 0.002, n),
        "HIGHVOL": base * 2.0 + rng.normal(0, 0.02, n),
        "MID": rng.normal(0, 0.01, n),
    }, index=idx)


W = pd.Series({"LOWVOL": 0.6, "HIGHVOL": 0.2, "MID": 0.2})


def test_return_contributions_reconcile() -> None:
    r = _returns()
    c = return_contributions(r, W)
    p = calculate_portfolio_returns(r, W)
    assert contribution_reconciliation_error(c, p) < 1e-15
    assert c.iloc[0]["HIGHVOL"] == pytest.approx(0.2 * r.iloc[0]["HIGHVOL"])


def test_volatility_contributions_sum_to_portfolio_vol() -> None:
    r = _returns()
    vc = volatility_contributions(r, W, 252)
    assert vc.table["vol_contribution_daily"].sum() == pytest.approx(vc.portfolio_vol_daily)
    assert vc.table["vol_contribution_share"].sum() == pytest.approx(1.0)
    p = calculate_portfolio_returns(r, W)
    assert vc.portfolio_vol_daily == pytest.approx(p.std(ddof=1))
    assert vc.portfolio_vol_annualized == pytest.approx(p.std(ddof=1) * np.sqrt(252))


def test_largest_weight_is_not_necessarily_largest_risk() -> None:
    vc = volatility_contributions(_returns(), W)
    t = vc.table
    assert t["weight"].idxmax() == "LOWVOL"
    assert t["vol_contribution_daily"].idxmax() == "HIGHVOL"


def test_extreme_loss_dates_are_worst_portfolio_days() -> None:
    r = _returns()
    out = extreme_loss_contributions(r, W, 5)
    p = calculate_portfolio_returns(r, W)
    assert list(out.index) == list(p.nsmallest(5).index)
    np.testing.assert_allclose(out[list(W.index)].sum(axis=1), out["portfolio_return"])


def test_full_diagnostics_bundle() -> None:
    res = calculate_asset_risk_contributions(_returns(), W, 252, 3)
    assert {"summary", "correlation", "extreme_losses", "volatility_contributions"} <= set(res)
    assert res["reconciliation_error"] < 1e-15
    assert res["correlation"].shape == (3, 3)


def test_contributions_reject_missing() -> None:
    r = _returns(10)
    r.iloc[3, 0] = np.nan
    with pytest.raises(MissingDataError):
        return_contributions(r, W)


# --- Limits ---------------------------------------------------------------------------------

def _loss(value: float, horizon: int = 5) -> MetricObservation:
    return MetricObservation("cumulative_loss", value, "fraction_of_initial_value", horizon, source="t")


@pytest.mark.parametrize("loss,breached", [(0.08, False), (0.12, True), (0.10, True), (-0.05, False)])
def test_h_loss_limit(loss, breached) -> None:
    res = evaluate_limit(LOSS_LIMIT, _loss(loss))
    assert res.breached is breached
    assert res.excess == pytest.approx(loss - 0.10)
    assert res.status == ("breach" if breached else "within_limit")


def test_equality_rule_is_configurable() -> None:
    strict = RiskLimitSpec("x", "cumulative_loss", "fraction_of_initial_value", 5, 0.10, ">")
    assert evaluate_limit(strict, _loss(0.10)).breached is False
    assert evaluate_limit(strict, _loss(0.1000001)).breached is True


def test_tolerance_absorbs_float_noise_at_equality() -> None:
    computed = -(0.3 * -0.1 + 0.7 * -0.1)  # 0.09999999999999999 in binary floating point
    assert computed < 0.10
    assert is_breach(computed, 0.10, ">=")
    assert not is_breach(0.0999, 0.10, ">=")


@pytest.mark.parametrize("obs", [
    _loss(0.5, horizon=1),
    MetricObservation("max_drawdown", 0.5, "fraction_of_peak_value", 5),
    MetricObservation("cumulative_loss", 0.5, "percent", 5),
])
def test_incompatible_observations_never_breach(obs) -> None:
    with pytest.raises(IncompatibleMetricError):
        evaluate_limit(LOSS_LIMIT, obs)
    res = evaluate_risk_limits([LOSS_LIMIT], [obs])
    assert len(res) == 1 and res[0].status == "not_evaluated" and res[0].breached is None


def test_var_limit_requires_matching_confidence() -> None:
    spec = RiskLimitSpec("v", "historical_var", "fraction_of_portfolio_value", 1, 0.03, ">=", confidence=0.95)
    obs99 = MetricObservation("historical_var", 0.05, "fraction_of_portfolio_value", 1, confidence=0.99)
    obs95 = MetricObservation("historical_var", 0.02, "fraction_of_portfolio_value", 1, confidence=0.95)
    res = evaluate_risk_limits([spec], [obs99, obs95])
    assert len(res) == 1 and res[0].confidence == 0.95 and res[0].breached is False
