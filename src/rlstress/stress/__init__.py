"""Deterministic stress testing: direct asset-return shocks and historical replays (Stage 3).

These evaluate user-specified scenarios; adaptive scenario discovery is Stage 4+.
"""

from rlstress.stress.deterministic import StressResult, apply_asset_return_shocks, validate_shocks
from rlstress.stress.historical_replay import (
    evaluate_return_path,
    replay_historical_scenario,
    worst_historical_windows,
)

__all__ = [
    "StressResult", "apply_asset_return_shocks", "evaluate_return_path", "replay_historical_scenario",
    "validate_shocks", "worst_historical_windows",
]
