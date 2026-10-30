"""Scenario storage and analysis.

Persists discovered scenarios with their metrics so they can be ranked,
clustered and inspected later without re-running the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from rlstress.stress import StressResult
from rlstress.shocks import ShockScenario


@dataclass
class ScenarioRecord:
    """A discovered stress scenario together with its evaluation.

    Attributes:
        scenario_id: Unique identifier.
        portfolio_name: Portfolio the scenario was discovered against.
        scenario: The shock sequence.
        risk: Risk metrics of the portfolio under the scenario.
        breaches: Limit name to breach flag.
        plausibility: Score in ``[0, 1]``.
        source: How it was produced, e.g. ``"ppo"``, ``"random"``, ``"historical"``.
        created_at: ISO-8601 timestamp.
        extra: Additional metadata.
    """

    scenario_id: str
    portfolio_name: str
    scenario: ShockScenario
    risk: StressResult
    breaches: dict[str, bool]
    plausibility: float
    source: str
    created_at: str
    extra: dict[str, Any] = field(default_factory=dict)


class ScenarioStore:
    """File-backed store of :class:`ScenarioRecord` objects."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def save(self, record: ScenarioRecord) -> Path:
        """Write one record and return its file path."""
        raise NotImplementedError

    def load(self, scenario_id: str) -> ScenarioRecord:
        """Load a record by id."""
        raise NotImplementedError

    def list_ids(self) -> list[str]:
        """Ids of all stored scenarios."""
        raise NotImplementedError

    def to_frame(self) -> pd.DataFrame:
        """Summary table: one row per scenario with its metrics and breach flags."""
        raise NotImplementedError


def summarize_scenarios(records: Iterable[ScenarioRecord]) -> pd.DataFrame:
    """Descriptive statistics of severity, plausibility and breach rates."""
    raise NotImplementedError


def cluster_scenarios(records: Iterable[ScenarioRecord], n_clusters: int = 5) -> pd.Series:
    """Group scenarios by shock pattern (``scikit-learn`` imported lazily)."""
    raise NotImplementedError
