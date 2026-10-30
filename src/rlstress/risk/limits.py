"""Risk-limit evaluation with explicit metric, unit, horizon and breach rule.

A limit (:class:`~rlstress.config.RiskLimitSpec`) is evaluated only against an
observation of the same metric, unit, horizon and (for VaR/CVaR) confidence.
Observations are loss magnitudes (positive = loss). A limit with no compatible
observation is reported as ``not_evaluated``, never as a breach or a pass.

Breach rule ``">="``: breach if ``observed >= threshold - tolerance``;
``">"``: breach if ``observed > threshold + tolerance``. The tolerance (default
1e-12) only absorbs floating-point error at exact equality.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from rlstress.config import RISK_METRIC_UNITS, RiskLimitSpec
from rlstress.exceptions import IncompatibleMetricError

STATUS_BREACH = "breach"
STATUS_WITHIN = "within_limit"
STATUS_NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True)
class MetricObservation:
    """A measured risk metric ready for limit comparison.

    Attributes:
        metric: Metric name (see :data:`~rlstress.config.RISK_METRIC_UNITS`).
        value: Loss magnitude (positive = loss; a gain is negative).
        units: Unit of ``value``.
        horizon_days: Trading days covered.
        confidence: Confidence level for VaR/CVaR, else ``None``.
        start_date / end_date: Observations or scenario path used.
        source: What produced the value (window or scenario name).
    """

    metric: str
    value: float
    units: str
    horizon_days: int
    confidence: float | None = None
    start_date: str | None = None
    end_date: str | None = None
    source: str = ""


@dataclass(frozen=True)
class LimitResult:
    """Outcome of comparing one observation with one limit."""

    limit_name: str
    metric: str
    units: str
    horizon_days: int
    threshold: float
    breach_rule: str
    status: str
    breached: bool | None
    observed: float | None
    excess: float | None
    confidence: float | None
    start_date: str | None
    end_date: str | None
    source: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return asdict(self)


def compatibility(spec: RiskLimitSpec, obs: MetricObservation) -> str:
    """Empty string if compatible, else the reason they cannot be compared."""
    if obs.metric != spec.metric:
        return f"metric {obs.metric} != {spec.metric}"
    if obs.units != spec.units or obs.units != RISK_METRIC_UNITS.get(obs.metric):
        return f"units {obs.units} != {spec.units}"
    if obs.horizon_days != spec.horizon_days:
        return f"horizon {obs.horizon_days}d != {spec.horizon_days}d"
    if spec.confidence != obs.confidence:
        return f"confidence {obs.confidence} != {spec.confidence}"
    return ""


def is_breach(observed: float, threshold: float, rule: str, tolerance: float = 1e-12) -> bool:
    """Apply a breach rule (see module docstring)."""
    if rule == ">=":
        return observed >= threshold - tolerance
    if rule == ">":
        return observed > threshold + tolerance
    raise ValueError(f"unknown breach rule {rule!r}")


def evaluate_limit(spec: RiskLimitSpec, obs: MetricObservation, tolerance: float = 1e-12) -> LimitResult:
    """Compare one compatible observation with one limit.

    Raises:
        IncompatibleMetricError: If metric, units, horizon or confidence differ.
    """
    reason = compatibility(spec, obs)
    if reason:
        raise IncompatibleMetricError(f"limit {spec.name}: {reason}")
    breached = is_breach(obs.value, spec.threshold, spec.breach_rule, tolerance)
    return LimitResult(
        limit_name=spec.name, metric=spec.metric, units=spec.units, horizon_days=spec.horizon_days,
        threshold=spec.threshold, breach_rule=spec.breach_rule,
        status=STATUS_BREACH if breached else STATUS_WITHIN, breached=breached,
        observed=obs.value, excess=obs.value - spec.threshold, confidence=spec.confidence,
        start_date=obs.start_date, end_date=obs.end_date, source=obs.source,
    )


def evaluate_risk_limits(
    limits: Iterable[RiskLimitSpec],
    observations: Iterable[MetricObservation],
    tolerance: float = 1e-12,
) -> list[LimitResult]:
    """Evaluate every limit against every compatible observation.

    Limits without a compatible observation yield one ``not_evaluated`` result
    explaining why; they are never treated as breached or passed.
    """
    obs_list = list(observations)
    results: list[LimitResult] = []
    for spec in limits:
        matched = [o for o in obs_list if not compatibility(spec, o)]
        if not matched:
            results.append(LimitResult(
                limit_name=spec.name, metric=spec.metric, units=spec.units,
                horizon_days=spec.horizon_days, threshold=spec.threshold,
                breach_rule=spec.breach_rule, status=STATUS_NOT_EVALUATED, breached=None,
                observed=None, excess=None, confidence=spec.confidence, start_date=None,
                end_date=None, source="",
                reason="no observation with matching metric, units, horizon and confidence",
            ))
            continue
        results.extend(evaluate_limit(spec, o, tolerance) for o in matched)
    return results
