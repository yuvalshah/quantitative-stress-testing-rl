"""Mitigation evaluation.

Applies a candidate mitigation (e.g. reducing exposure, rebalancing) to a
portfolio and replays a stored scenario to measure how much risk it removes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rlstress.config import MitigationConfig, RiskConfig
from rlstress.portfolio import Portfolio
from rlstress.stress import StressResult
from rlstress.scenarios import ScenarioRecord


@dataclass(frozen=True)
class MitigationStrategy:
    """A parameterised portfolio adjustment."""

    name: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MitigationResult:
    """Before/after comparison of a mitigation under one scenario."""

    strategy: MitigationStrategy
    scenario_id: str
    before: StressResult
    after: StressResult
    breaches_before: dict[str, bool]
    breaches_after: dict[str, bool]
    turnover: float
    cost: float


def available_strategies(config: MitigationConfig) -> list[MitigationStrategy]:
    """Instantiate the strategies named in the ``[mitigation]`` config section."""
    raise NotImplementedError


def apply_mitigation(
    portfolio: Portfolio, strategy: MitigationStrategy, config: MitigationConfig
) -> Portfolio:
    """Return the adjusted portfolio, respecting ``config.max_turnover``."""
    raise NotImplementedError


def evaluate_mitigation(
    portfolio: Portfolio,
    strategy: MitigationStrategy,
    record: ScenarioRecord,
    limits: RiskConfig,
    config: MitigationConfig,
) -> MitigationResult:
    """Replay ``record`` on the original and mitigated portfolios and compare."""
    raise NotImplementedError
