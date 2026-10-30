"""Scenario recommendation.

Ranks discovered scenarios for a selected portfolio by a weighted combination of
severity, plausibility and portfolio relevance, then proposes mitigations to test.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from rlstress.config import MitigationConfig, RecommendationConfig, RiskConfig
from rlstress.mitigation import MitigationResult, MitigationStrategy
from rlstress.portfolio import Portfolio
from rlstress.scenarios import ScenarioRecord


def severity_score(record: ScenarioRecord, limits: RiskConfig) -> float:
    """Normalised severity in ``[0, 1]`` based on how far limits are exceeded."""
    raise NotImplementedError


def relevance_score(record: ScenarioRecord, portfolio: Portfolio) -> float:
    """Relevance in ``[0, 1]``: how strongly the scenario hits this portfolio's exposures."""
    raise NotImplementedError


def rank_scenarios(
    records: Sequence[ScenarioRecord],
    portfolio: Portfolio,
    limits: RiskConfig,
    config: RecommendationConfig,
) -> pd.DataFrame:
    """Top-``config.top_k`` scenarios with component and combined scores."""
    raise NotImplementedError


def recommend_mitigations(
    record: ScenarioRecord,
    portfolio: Portfolio,
    strategies: Sequence[MitigationStrategy],
    limits: RiskConfig,
    config: MitigationConfig,
) -> list[MitigationResult]:
    """Evaluate each strategy against ``record`` and order by risk reduction per unit cost."""
    raise NotImplementedError
