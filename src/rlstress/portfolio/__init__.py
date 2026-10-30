"""Portfolio configuration, validation, return aggregation and valuation (Stage 3)."""

from rlstress.portfolio.model import Portfolio
from rlstress.portfolio.returns import (
    AlignedReturns,
    apply_missing_data_policy,
    calculate_asset_returns,
    calculate_portfolio_returns,
    select_window,
)
from rlstress.portfolio.validation import (
    normalize_weights,
    validate_capital,
    validate_confidence,
    validate_horizon,
    validate_portfolio,
    validate_return_frame,
    validate_weights,
)
from rlstress.portfolio.valuation import (
    base_date_for,
    buy_and_hold_value,
    calculate_portfolio_value,
    value_consistency_error,
)

__all__ = [
    "AlignedReturns", "Portfolio", "apply_missing_data_policy", "base_date_for",
    "buy_and_hold_value", "calculate_asset_returns", "calculate_portfolio_returns",
    "calculate_portfolio_value", "normalize_weights", "select_window", "validate_capital",
    "validate_confidence", "validate_horizon", "validate_portfolio", "validate_return_frame",
    "validate_weights", "value_consistency_error",
]
