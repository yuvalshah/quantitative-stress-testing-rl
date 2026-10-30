"""Historical risk metrics, asset-level diagnostics and risk limits (Stage 3).

Conventions: daily simple returns; losses positive for VaR/CVaR/limits;
maximum drawdown reported negative; all values decimal fractions unless the
name ends in ``_usd``.
"""

from rlstress.risk.contributions import (
    VolatilityContributions,
    asset_correlation,
    calculate_asset_risk_contributions,
    contribution_reconciliation_error,
    extreme_loss_contributions,
    return_contributions,
    volatility_contributions,
)
from rlstress.risk.cvar import CVaRResult, calculate_historical_cvar
from rlstress.risk.drawdown import DrawdownResult, calculate_max_drawdown, drawdown_series
from rlstress.risk.limits import (
    LimitResult,
    MetricObservation,
    evaluate_limit,
    evaluate_risk_limits,
    is_breach,
)
from rlstress.risk.var import VaRResult, calculate_historical_var
from rlstress.risk.volatility import (
    VolatilityResult,
    asset_volatilities,
    calculate_historical_volatility,
    rolling_volatility_table,
)

__all__ = [
    "CVaRResult", "DrawdownResult", "LimitResult", "MetricObservation", "VaRResult",
    "VolatilityContributions", "VolatilityResult", "asset_correlation", "asset_volatilities",
    "calculate_asset_risk_contributions", "calculate_historical_cvar", "calculate_historical_var",
    "calculate_historical_volatility", "calculate_max_drawdown", "contribution_reconciliation_error",
    "drawdown_series", "evaluate_limit", "evaluate_risk_limits", "extreme_loss_contributions",
    "is_breach", "return_contributions", "rolling_volatility_table", "volatility_contributions",
]
