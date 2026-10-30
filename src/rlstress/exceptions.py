"""Exception types shared by the portfolio, risk and stress modules."""

from __future__ import annotations


class RiskEngineError(ValueError):
    """Base class for invalid inputs to the risk engine."""


class PortfolioValidationError(RiskEngineError):
    """Portfolio weights, capital or policy are invalid."""


class ReturnDataError(RiskEngineError):
    """Return data have a bad schema, dates or values."""


class MissingDataError(ReturnDataError):
    """Required return observations are missing under the configured policy."""


class InsufficientDataError(RiskEngineError):
    """Too few observations for the requested estimator."""


class InvalidShockError(RiskEngineError):
    """A stress shock is impossible or does not match the portfolio."""


class IncompatibleMetricError(RiskEngineError):
    """A risk limit was compared with a metric of another type, unit, horizon or confidence."""
