"""Asset-return selection, missing-data policy and portfolio-return aggregation.

Asset returns come from Stage 2 (``{TICKER}_return``: simple returns of adjusted
close, ``P_t / P_(t-1) - 1``, computed on the market calendar, never filled and
undefined across gaps; see :mod:`rlstress.data.returns`). They are selected,
not recomputed, so the Stage 2 transformation stays the single source of truth.

Portfolio return under daily constant-mix rebalancing (weights reset to target
every period, no transaction costs, taxes or cash flows):

    r_p,t = sum_i w_i * r_i,t

A buy-and-hold portfolio, whose weights drift with prices, is valued
separately by :func:`rlstress.portfolio.valuation.buy_and_hold_value`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from rlstress.exceptions import MissingDataError, ReturnDataError
from rlstress.portfolio.validation import validate_return_frame

logger = logging.getLogger(__name__)


@dataclass
class AlignedReturns:
    """Complete-case asset returns plus a record of what the missing-data policy excluded."""

    returns: pd.DataFrame
    policy: str
    n_input_rows: int
    excluded_dates: list[str] = field(default_factory=list)

    @property
    def start_date(self) -> str | None:
        """First date with a complete return vector."""
        return self.returns.index.min().date().isoformat() if len(self.returns) else None

    @property
    def end_date(self) -> str | None:
        """Last date with a complete return vector."""
        return self.returns.index.max().date().isoformat() if len(self.returns) else None

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable description."""
        return {
            "missing_data_policy": self.policy,
            "n_input_rows": self.n_input_rows,
            "n_complete_rows": int(len(self.returns)),
            "n_excluded_rows": len(self.excluded_dates),
            "excluded_dates": self.excluded_dates[:50],
            "start_date": self.start_date,
            "end_date": self.end_date,
        }


def calculate_asset_returns(
    source: pd.DataFrame,
    tickers: Sequence[str],
    column_template: str = "{ticker}_return",
) -> pd.DataFrame:
    """Select the Stage 2 return columns of ``tickers`` into a frame with ticker columns.

    Missing values are preserved. Raises :class:`ReturnDataError` for a bad
    index, duplicate/unsorted dates, absent columns or returns below -100%.
    """
    columns = [column_template.format(ticker=t) for t in tickers]
    validate_return_frame(source, columns)
    out = source[columns].astype(float)
    out.columns = list(tickers)
    return out


def select_window(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """Rows with ``start <= date <= end`` (inclusive; ``None`` = open-ended)."""
    mask = np.ones(len(frame), dtype=bool)
    if start is not None:
        mask &= frame.index >= pd.Timestamp(start)
    if end is not None:
        mask &= frame.index <= pd.Timestamp(end)
    return frame[mask]


def apply_missing_data_policy(asset_returns: pd.DataFrame, policy: str = "complete_case") -> AlignedReturns:
    """Apply the missing-data policy. Returns are never zero-filled.

    * ``complete_case``: drop dates where any holding's return is missing and
      record them.
    * ``raise``: raise :class:`MissingDataError` if any return is missing.
    """
    incomplete = asset_returns.isna().any(axis=1)
    excluded = [d.date().isoformat() for d in asset_returns.index[incomplete]]
    if policy == "raise":
        if excluded:
            raise MissingDataError(f"{len(excluded)} date(s) with missing holding returns, e.g. {excluded[:5]}")
    elif policy != "complete_case":
        raise ReturnDataError(f"unknown missing-data policy {policy!r}")
    if excluded:
        logger.info("complete_case: excluded %d date(s) with missing holding returns", len(excluded))
    return AlignedReturns(asset_returns[~incomplete], policy, int(len(asset_returns)), excluded)


def calculate_portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Daily constant-mix portfolio returns ``r_p,t = sum_i w_i r_i,t``.

    Args:
        asset_returns: Complete-case returns with one column per holding.
        weights: Target weights indexed by ticker.

    Raises:
        ReturnDataError: If holdings and columns differ.
        MissingDataError: If any return is missing (apply the policy first).
    """
    if set(asset_returns.columns) != set(weights.index):
        raise ReturnDataError(
            f"return columns {sorted(asset_returns.columns)} do not match holdings {sorted(weights.index)}"
        )
    ordered = asset_returns[list(weights.index)]
    if ordered.isna().any().any():
        raise MissingDataError("portfolio returns need a complete return vector on every date")
    validate_return_frame(ordered)
    values = ordered.to_numpy(dtype=float) @ weights.to_numpy(dtype=float)
    return pd.Series(values, index=ordered.index, name="portfolio_return")
