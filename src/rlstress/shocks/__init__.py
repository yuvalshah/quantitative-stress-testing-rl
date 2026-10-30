"""Stage 4 market-shock model: factor model + multivariate Student-t, plausibility, simulator."""

from rlstress.shocks.factor_model import ShockModel, fit_shock_model, training_matrix
from rlstress.shocks.plausibility import (
    Plausibility,
    factor_plausibility,
    historical_path_percentile,
    implausibility_credit_fraction,
    path_log_tail_probability,
    path_tail_probability,
    plausibility_log_threshold,
)
from rlstress.shocks.simulator import (
    InvalidActionError,
    InvalidSimulationError,
    ShockScenario,
    ShockSimulator,
    SimulatedStep,
)

__all__ = [
    "InvalidActionError", "InvalidSimulationError", "Plausibility", "ShockModel", "ShockScenario",
    "ShockSimulator", "SimulatedStep", "factor_plausibility", "fit_shock_model",
    "historical_path_percentile", "implausibility_credit_fraction", "path_log_tail_probability",
    "path_tail_probability", "plausibility_log_threshold", "training_matrix",
]
