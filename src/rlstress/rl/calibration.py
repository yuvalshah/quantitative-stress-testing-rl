"""Calibration of plausibility thresholds against real market weeks.

Threshold semantics (identical to Stage 4): ``min_plausibility = q`` accepts a
simulated path iff its *historical percentile* is ``>= q``, where the
historical percentile is the share of actual TRAINING windows of the same
length whose Fisher-combined model path probability is at most the path's
(i.e. at least as extreme). It is a percentile of extremeness, not a tail
probability.

Validation calibration: the same percentile is computed for every actual
5-day window of the validation period (shock model and reference windows
remain training-only). If the model and threshold were calibrated out of
sample, about a share ``q`` of real validation weeks would be rejected.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from rlstress.portfolio import select_window
from rlstress.shocks import ShockModel, factor_plausibility, historical_path_percentile


def real_window_percentiles(model: ShockModel, source: pd.DataFrame, start: str, end: str,
                            horizon: int) -> pd.Series:
    """Historical percentile of every actual ``horizon``-day factor window in ``[start, end]``.

    Indexed by window start date; windows touching a day with a missing factor are skipped.
    """
    f = select_window(source[list(model.factors)], start, end)
    surprisal = pd.Series([factor_plausibility(model, row).surprisal if np.all(np.isfinite(row)) else np.nan
                           for row in f.to_numpy()], index=f.index)
    out = {}
    for i in range(len(surprisal) - horizon + 1):
        window = surprisal.iloc[i:i + horizon]
        if window.isna().any():
            continue
        out[window.index[0]] = historical_path_percentile(model, window.tolist())
    return pd.Series(out, name="historical_percentile", dtype=float)


def threshold_calibration(model: ShockModel, source: pd.DataFrame, windows: dict[str, tuple[str, str]],
                          thresholds: Sequence[float], horizon: int) -> dict[str, Any]:
    """Share of real windows rejected at each threshold, per period (train, validation)."""
    out: dict[str, Any] = {"semantics": "reject a path iff historical_percentile < threshold "
                                        "(percentile of extremeness among real training windows)"}
    for name, (start, end) in windows.items():
        pct = real_window_percentiles(model, source, start, end, horizon)
        out[name] = {
            "n_windows": int(len(pct)),
            "first_window": pct.index.min().date().isoformat() if len(pct) else None,
            "last_window": pct.index.max().date().isoformat() if len(pct) else None,
            "rejected_share": {str(q): float((pct < q).mean()) for q in thresholds},
        }
    if "validation" in out:
        out["validation_calibration_ratio"] = {
            str(q): (out["validation"]["rejected_share"][str(q)] / q) for q in thresholds}
    return out


def select_threshold(calibration: dict[str, Any], thresholds: Sequence[float], default: float) -> dict[str, Any]:
    """Choose the threshold whose validation rejection share is closest to nominal (relative error).

    The threshold is a plausibility definition, so it is selected for
    calibration on real validation weeks, never for breach counts. Ties
    (within 0.05 relative error) go to ``default``.
    """
    errors = {q: abs(calibration["validation"]["rejected_share"][str(q)] - q) / q for q in thresholds}
    best = min(errors, key=errors.get)
    if default in errors and errors[default] - errors[best] <= 0.05:
        best = default
    return {"selected": best, "relative_calibration_error": {str(q): e for q, e in errors.items()},
            "rule": "min |validation rejected share - q| / q; ties within 0.05 keep the default"}
