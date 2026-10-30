"""Replay of actual historical return windows against the portfolio.

A replay takes ``horizon_days`` consecutive trading days of actual asset
returns (Stage 2, market calendar) starting at the first trading date on or
after ``start_date``, applies daily constant-mix rebalancing, and reports the
value path, cumulative loss and path drawdown. Every day in the window must have
a complete return vector, otherwise the replay is refused. Historical episodes
are a small sample of what can happen; they are not RL-generated scenarios.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from rlstress.config import RiskLimitSpec
from rlstress.exceptions import InsufficientDataError, MissingDataError, ReturnDataError
from rlstress.portfolio import Portfolio, calculate_portfolio_returns, calculate_portfolio_value, validate_horizon
from rlstress.portfolio.valuation import base_date_for
from rlstress.risk.drawdown import calculate_max_drawdown
from rlstress.stress.deterministic import StressResult

REPLAY_NOTE = "historical replay: actual daily returns, weights reset to target daily (constant mix), no costs"


def replay_historical_scenario(
    portfolio: Portfolio,
    asset_returns: pd.DataFrame,
    start_date: str,
    horizon_days: int,
    limits: Iterable[RiskLimitSpec] = (),
    name: str = "historical_replay",
    description: str = "",
    tolerance: float = 1e-12,
) -> StressResult:
    """Replay ``horizon_days`` trading days of actual returns from ``start_date``.

    Args:
        portfolio: Portfolio to value.
        asset_returns: Asset returns on the market calendar (ticker columns, may
            contain missing values elsewhere).
        start_date: First trading date (or the next trading date after it).

    Raises:
        InsufficientDataError: Fewer than ``horizon_days`` dates available.
        MissingDataError: Any return missing inside the window.
    """
    horizon = validate_horizon(horizon_days)
    frame = asset_returns[list(portfolio.tickers)]
    window = frame[frame.index >= pd.Timestamp(start_date)].iloc[:horizon]
    if len(window) < horizon:
        raise InsufficientDataError(
            f"replay {name}: only {len(window)} trading days from {start_date}, need {horizon}"
        )
    if window.isna().any().any():
        raise MissingDataError(f"replay {name}: missing returns inside the window")
    try:
        base = base_date_for(pd.DatetimeIndex(frame.index), window.index[0])
    except ReturnDataError:
        base = window.index[0] - pd.Timedelta(days=1)  # no prior row: label start as the day before
    return evaluate_return_path(portfolio, window, base, limits, name, "historical_replay",
                                description, tolerance, {"rebalancing": REPLAY_NOTE})


def evaluate_return_path(
    portfolio: Portfolio,
    path_returns: pd.DataFrame,
    base_date: pd.Timestamp,
    limits: Iterable[RiskLimitSpec] = (),
    name: str = "return_path",
    scenario_type: str = "return_path",
    description: str = "",
    tolerance: float = 1e-12,
    assumptions: dict | None = None,
) -> StressResult:
    """Value a multi-day path of asset returns with daily constant-mix rebalancing.

    Shared by historical replays and the Stage 4 simulator: the path is valued
    with the Stage 3 portfolio-return, valuation and drawdown functions, and the
    horizon equals the number of rows.

    Args:
        path_returns: Complete daily asset returns (ticker columns), one row per day.
        base_date: Date label of the starting value (before the first row).
    """
    horizon = validate_horizon(len(path_returns))
    window = path_returns[list(portfolio.tickers)]
    r_p = calculate_portfolio_returns(window, portfolio.weights)
    values = calculate_portfolio_value(r_p, portfolio.initial_capital, base_date)
    dd = calculate_max_drawdown(values)
    v0, v1 = float(values.iloc[0]), float(values.iloc[-1])
    cum = v1 / v0 - 1.0
    contrib_sum = (window * portfolio.weights).sum()
    result = StressResult(
        scenario_name=name,
        scenario_type=scenario_type,
        horizon_days=horizon,
        initial_value=v0,
        final_value=v1,
        portfolio_return=cum,
        loss_usd=v0 - v1,
        loss_pct=-cum,
        max_drawdown=dd.max_drawdown,
        asset_contributions=[
            {"ticker": t, "weight": float(portfolio.weights[t]),
             "asset_cumulative_return": float(np.prod(1 + window[t].to_numpy()) - 1),
             "sum_of_daily_contributions": float(contrib_sum[t])}
            for t in portfolio.tickers
        ],
        start_date=window.index[0].date().isoformat(),
        end_date=window.index[-1].date().isoformat(),
        value_path={d.date().isoformat(): float(v) for d, v in values.items()},
        assumptions={**(assumptions or {}), "base_date": pd.Timestamp(base_date).date().isoformat(),
                     "note": "sum_of_daily_contributions does not add up to the compounded return"},
        description=description,
    )
    result.evaluate_limits(limits, tolerance)
    return result


def worst_historical_windows(
    asset_returns: pd.DataFrame, weights: pd.Series, horizon_days: int, top_n: int
) -> pd.DataFrame:
    """The ``top_n`` worst non-overlapping ``horizon_days`` compounded constant-mix returns.

    Windows touching a date with an incomplete return vector are skipped (never
    filled). Descriptive only: the search uses whatever period it is given.
    """
    horizon = validate_horizon(horizon_days)
    frame = asset_returns[list(weights.index)]
    complete = frame.notna().all(axis=1)
    daily = pd.Series(frame.fillna(0.0).to_numpy() @ weights.to_numpy(), index=frame.index)
    daily[~complete] = np.nan  # incomplete days invalidate any window that contains them
    log_growth = np.log1p(daily)
    cum = np.expm1(log_growth.rolling(horizon, min_periods=horizon).sum())
    order = cum.dropna().sort_values()
    positions = {d: i for i, d in enumerate(frame.index)}
    taken: list[tuple[int, int]] = []
    rows = []
    for end_date, value in order.items():
        end = positions[end_date]
        start = end - horizon + 1
        if any(not (end < a or start > b) for a, b in taken):
            continue
        taken.append((start, end))
        rows.append({"start_date": frame.index[start].date().isoformat(),
                     "end_date": end_date.date().isoformat(),
                     "cumulative_return": float(value)})
        if len(rows) == top_n:
            break
    return pd.DataFrame(rows, columns=["start_date", "end_date", "cumulative_return"])
