"""Maximum drawdown of a portfolio value path.

``Peak_t = max_{s <= t} V_s``, ``DD_t = V_t / Peak_t - 1``, ``MDD = min_t DD_t``
(reported as a negative fraction; 0 means no decline). Peak = the most recent
date before the trough at which the running maximum was set; recovery = first
date after the trough with ``V >= peak value``. Drawdown is a path statistic
and is unrelated to one-day losses or VaR.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from rlstress.exceptions import InsufficientDataError, MissingDataError

CONVENTION = "MDD = min(V_t / running_max(V) - 1); negative fraction"


@dataclass(frozen=True)
class DrawdownResult:
    """Maximum drawdown and its dates."""

    max_drawdown: float
    peak_date: str | None
    trough_date: str | None
    peak_value: float | None
    trough_value: float | None
    recovery_date: str | None
    recovered: bool | None
    n_observations: int
    start_date: str
    end_date: str
    convention: str = CONVENTION

    @property
    def magnitude(self) -> float:
        """Positive size of the drawdown (``-max_drawdown``)."""
        return -self.max_drawdown

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return asdict(self)


def drawdown_series(values: pd.Series) -> pd.Series:
    """``V_t / running max - 1``."""
    return values / values.cummax() - 1.0


def calculate_max_drawdown(values: pd.Series) -> DrawdownResult:
    """Maximum drawdown with peak, trough and recovery information.

    ``recovered`` is ``None`` when there was no drawdown at all.
    """
    if values.isna().any():
        raise MissingDataError("value path contains missing values")
    if len(values) < 2:
        raise InsufficientDataError("drawdown needs at least two values")
    if (values <= 0).any():
        raise MissingDataError("value path must stay positive")
    dd = drawdown_series(values)
    iso = lambda ts: pd.Timestamp(ts).date().isoformat()  # noqa: E731
    mdd = float(dd.min())
    start, end = iso(values.index[0]), iso(values.index[-1])
    if mdd >= 0:
        return DrawdownResult(0.0, None, None, None, None, None, None, len(values), start, end)
    trough_pos = int(dd.to_numpy().argmin())
    before = values.iloc[: trough_pos + 1]
    peak_value = float(before.max())
    peak_pos = int(max(i for i, v in enumerate(before.to_numpy()) if v == peak_value))
    after = values.iloc[trough_pos + 1:]
    rec = after[after >= peak_value]
    recovery = iso(rec.index[0]) if len(rec) else None
    return DrawdownResult(
        max_drawdown=mdd,
        peak_date=iso(values.index[peak_pos]),
        trough_date=iso(values.index[trough_pos]),
        peak_value=peak_value,
        trough_value=float(values.iloc[trough_pos]),
        recovery_date=recovery,
        recovered=recovery is not None,
        n_observations=len(values),
        start_date=start,
        end_date=end,
    )
