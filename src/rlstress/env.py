"""Gymnasium stress-testing environment (Stage 4).

An episode is one stress path of ``simulation.horizon_days`` trading days (5)
starting from the actual market state on an anchor date drawn from the anchor
split. Each step the agent chooses bounded factor shocks; the simulator turns
them into stock returns (factor model + conditional Student-t residuals); the
Stage 3 engine values the path and checks the path limits.

Termination: a breach of a path limit with the episode horizon (5-day cumulative
loss or 5-day drawdown), or the horizon. ``truncated`` is always ``False``
(the horizon is part of the task and the remaining horizon is observed).

Limit timing: path drawdown can only grow as the path lengthens, so a drawdown
of at least the threshold on a prefix already guarantees a breach over the full
horizon and is checked every step. Cumulative loss is checked at the horizon
only (it can recover). With equal 10% thresholds any loss breach is also a
drawdown breach.

Reward per step (components reported separately in ``info``)::

    breach        = breach_reward                if a VALID breach happens this step
                  = -implausible_breach_penalty  if an IMPLAUSIBLE breach happens this step (once per episode)
    plausibility_credit = kappa * breach_reward * L*_t / ell_t   on that IMPLAUSIBLE breach (once per episode)
    severity      = severity_weight * (DD_t - DD_(t-1))       (path drawdown magnitude)
    plausibility  = -plausibility_penalty * min(-ln p_t, surprisal_cap)
    magnitude     = -magnitude_penalty * mean(a_t^2)
    step          = -step_penalty
    invalid       = -invalid_penalty                if the simulator cannot produce valid returns

A breach is VALID only if the path is no more extreme than the most extreme
``simulation.min_plausibility`` share (default 1%) of actual training windows
of the same length, ranked by the Fisher-combined model tail probability
(:func:`~rlstress.shocks.historical_path_percentile`). An implausible breach
still ends the episode but earns no breach reward; with
``implausible_breach_penalty > 0`` it is also penalised, so an implausible crash
is not a cheap early exit from the per-step penalties (the Stage 5
"implausible-crash trap"). The default 0 reproduces the original reward.

Graded credit (``implausible_breach_credit = kappa``, default 0): an implausible
breach also earns ``kappa * breach_reward * L*_t / ell_t``, where ``ell_t`` is the
path log tail probability after ``t`` days and ``L*_t`` the value at which the
path would become plausible (exactly equivalent to the percentile gate). The
credit is continuous, increases as the path approaches plausibility, is bounded
by ``kappa * breach_reward`` and tends to 0 for ever more extreme paths, so it
gives a learning signal out of the implausible-crash region while a valid
breach still pays more (for the default weights every valid episode beats every
implausible one when kappa < 0.68; the config requires kappa < 1).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from rlstress.config import (
    RISK_METRIC_UNITS,
    EnvironmentConfig,
    ProjectConfig,
    RiskConfig,
    SimulationConfig,
)
from rlstress.portfolio import Portfolio, select_window
from rlstress.risk.limits import LimitResult, MetricObservation, evaluate_risk_limits
from rlstress.shocks import (
    InvalidSimulationError,
    ShockModel,
    ShockScenario,
    ShockSimulator,
    fit_shock_model,
    historical_path_percentile,
    implausibility_credit_fraction,
    path_log_tail_probability,
    plausibility_log_threshold,
)
from rlstress.stress import StressResult, evaluate_return_path
from rlstress.stress.deterministic import PATH_METRICS

SCENARIO_TYPE = "simulated_factor_shock"


class StressTestEnv(gym.Env):
    """Environment in which an agent searches for plausible limit-breaching factor-shock paths.

    Action space: ``Box(-1, 1, (k,))``, one entry per factor (see
    :class:`~rlstress.shocks.ShockSimulator`). Observation: see :attr:`observation_names`.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        portfolio: Portfolio,
        simulator: ShockSimulator,
        risk: RiskConfig,
        env_config: EnvironmentConfig,
        sim_config: SimulationConfig,
        market_states: pd.DataFrame,
        state_mean: pd.Series,
        state_sd: pd.Series,
    ) -> None:
        super().__init__()
        if tuple(simulator.model.tickers) != portfolio.tickers:
            raise ValueError("shock model tickers must match portfolio holdings in order")
        self.portfolio = portfolio
        self.simulator = simulator
        self.risk = risk
        self.config = env_config
        self.sim_config = sim_config
        self.horizon = sim_config.horizon_days
        self.path_limits = [l for l in risk.limits
                            if l.metric in PATH_METRICS and l.horizon_days == self.horizon]
        if not self.path_limits:
            raise ValueError(f"no cumulative_loss/max_drawdown limit with horizon {self.horizon} days")
        if market_states.empty or market_states.isna().any().any():
            raise ValueError("market_states must be non-empty and complete")
        self.market_states = market_states
        self.state_mean = state_mean[market_states.columns]
        self.state_sd = state_sd[market_states.columns].replace(0, 1.0)
        model = simulator.model
        self.observation_names = (
            ["remaining_fraction"]
            + [f"cum_{f}" for f in model.factors] + [f"last_{f}" for f in model.factors]
            + [f"cum_{t}_return" for t in model.tickers]
            + ["value_ratio", "cumulative_loss", "path_drawdown", "last_portfolio_return",
               "cumulative_surprisal", "path_log_tail_probability", "path_historical_percentile"]
            + [f"state_{c}" for c in market_states.columns]
        )
        self.log_thresholds = {t: plausibility_log_threshold(model, t, sim_config.min_plausibility)
                               for t in range(1, self.horizon + 1)}
        self.action_space = spaces.Box(-1.0, 1.0, shape=(model.k,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(self.observation_names),),
                                            dtype=np.float32)
        self._t = 0
        self._done = True

    # -- episode state -------------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a path at an anchor date (``options["anchor_date"]`` or drawn with the env RNG)."""
        super().reset(seed=seed)
        options = options or {}
        if "anchor_date" in options:
            self.anchor = pd.Timestamp(options["anchor_date"])
            if self.anchor not in self.market_states.index:
                raise ValueError(f"anchor date {self.anchor.date()} not in the anchor window")
        else:
            self.anchor = self.market_states.index[int(self.np_random.integers(len(self.market_states)))]
        self._state = ((self.market_states.loc[self.anchor] - self.state_mean) / self.state_sd).to_numpy()
        k, n = self.simulator.model.k, self.portfolio.n_assets
        self._factor_path = np.zeros((0, k))
        self._return_path = np.zeros((0, n))
        self._surprisals: list[float] = []
        self._t = 0
        self._dd = 0.0
        self._last_rp = 0.0
        self._result: StressResult | None = None
        self._breach_paid = False
        self._implausible_charged = False
        self._done = False
        return self._observation(), {"anchor_date": self.anchor.date().isoformat(), "step": 0}

    def _observation(self) -> np.ndarray:
        cum_f = np.prod(1 + self._factor_path, axis=0) - 1 if len(self._factor_path) else np.zeros(self.simulator.model.k)
        last_f = self._factor_path[-1] if len(self._factor_path) else np.zeros(self.simulator.model.k)
        cum_r = np.prod(1 + self._return_path, axis=0) - 1 if len(self._return_path) else np.zeros(self.portfolio.n_assets)
        r = self._result
        value_ratio = r.final_value / r.initial_value if r else 1.0
        loss = r.loss_pct if r else 0.0
        obs = np.concatenate([
            [(self.horizon - self._t) / self.horizon], cum_f, last_f, cum_r,
            [value_ratio, loss, self._dd, self._last_rp, float(np.sum(self._surprisals)),
             path_log_tail_probability(self._surprisals),
             historical_path_percentile(self.simulator.model, self._surprisals)],
            self._state,
        ]).astype(np.float32)
        return obs

    def _path_frame(self) -> pd.DataFrame:
        # Step dates are labels: the next business days after the anchor (simulated, not historical).
        dates = pd.bdate_range(self.anchor + pd.offsets.BDay(1), periods=len(self._return_path))
        return pd.DataFrame(self._return_path, index=dates, columns=list(self.portfolio.tickers))

    def _limit_observations(self, result: StressResult) -> list[MetricObservation]:
        common = dict(horizon_days=self.horizon, start_date=result.start_date, end_date=result.end_date,
                      source="simulated path")
        obs = [MetricObservation("max_drawdown", -result.max_drawdown,
                                 RISK_METRIC_UNITS["max_drawdown"], **common)]
        if self._t == self.horizon:
            obs.append(MetricObservation("cumulative_loss", result.loss_pct,
                                         RISK_METRIC_UNITS["cumulative_loss"], **common))
        return obs

    # -- step ----------------------------------------------------------------------------------

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one day of factor shocks. Invalid actions raise ``InvalidActionError``."""
        if self._done:
            raise RuntimeError("episode finished; call reset()")
        cfg = self.config
        try:
            sim = self.simulator.step(action, self.np_random)
        except InvalidSimulationError as exc:
            self._done = True
            comps = {"breach": 0.0, "severity": 0.0, "plausibility": 0.0, "magnitude": 0.0,
                     "step": -cfg.step_penalty, "invalid": -cfg.invalid_penalty, "plausibility_credit": 0.0}
            return self._observation(), float(sum(comps.values())), True, False, {
                "invalid": True, "error": str(exc), "reward_components": comps, "step": self._t}
        self._t += 1
        self._factor_path = np.vstack([self._factor_path, sim.factor_shock])
        self._return_path = np.vstack([self._return_path, sim.stock_returns])
        self._surprisals.append(sim.plausibility.surprisal)
        result = evaluate_return_path(self.portfolio, self._path_frame(), self.anchor, (),
                                      "simulated_episode", SCENARIO_TYPE)
        self._result = result
        limit_results: list[LimitResult] = evaluate_risk_limits(
            self.path_limits, self._limit_observations(result), self.risk.comparison_tolerance)
        breached = any(r.breached for r in limit_results)
        path_log_p = path_log_tail_probability(self._surprisals)
        percentile = historical_path_percentile(self.simulator.model, self._surprisals)
        plausible = percentile >= self.sim_config.min_plausibility
        valid_breach = breached and plausible and not self._breach_paid
        implausible_now = breached and not plausible and not self._implausible_charged
        new_dd = max(0.0, -result.max_drawdown)
        comps = {
            "breach": (cfg.breach_reward if valid_breach
                       else -cfg.implausible_breach_penalty if implausible_now else 0.0),
            "severity": cfg.severity_weight * (new_dd - self._dd),
            "plausibility": -cfg.plausibility_penalty * min(sim.plausibility.surprisal, self.sim_config.surprisal_cap),
            "magnitude": -cfg.magnitude_penalty * float(np.mean(sim.action ** 2)),
            "step": -cfg.step_penalty,
            "invalid": 0.0,
            "plausibility_credit": (cfg.implausible_breach_credit * cfg.breach_reward
                                    * implausibility_credit_fraction(path_log_p, self.log_thresholds[self._t])
                                    if implausible_now else 0.0),
        }
        self._breach_paid |= valid_breach
        self._implausible_charged |= implausible_now
        self._dd = new_dd
        prev_value = result.value_path[list(result.value_path)[-2]]
        self._last_rp = result.final_value / prev_value - 1.0
        terminated = (breached and cfg.terminate_on_breach) or self._t >= self.horizon
        self._done = terminated
        info = {
            "step": self._t,
            "anchor_date": self.anchor.date().isoformat(),
            "action": sim.action.tolist(),
            "factor_shock": dict(zip(self.simulator.model.factors, sim.factor_shock.tolist())),
            "stock_returns": dict(zip(self.portfolio.tickers, sim.stock_returns.tolist())),
            "portfolio_value": result.final_value,
            "cumulative_loss": result.loss_pct,
            "path_drawdown": new_dd,
            "plausibility": sim.plausibility.to_dict(),
            "path_log_tail_probability": path_log_p,
            "path_historical_percentile": percentile,
            "breached": breached,
            "valid_breach": breached and plausible,
            "implausible_breach": breached and not plausible,
            "invalid": False,
            "residual_draws": sim.residual_draws,
            "limit_results": [r.to_dict() for r in limit_results],
            "reward_components": comps,
        }
        return self._observation(), float(sum(comps.values())), terminated, False, info

    # -- inspection ----------------------------------------------------------------------------

    def current_result(self) -> StressResult | None:
        """Stage 3 valuation of the path so far, with path limits judged as in :meth:`step`."""
        if not len(self._return_path):
            return None
        result = evaluate_return_path(self.portfolio, self._path_frame(), self.anchor, (),
                                      "simulated_episode", SCENARIO_TYPE)
        # Same limit timing as step(): prefix drawdown always, cumulative loss at the horizon only.
        result.limit_results = evaluate_risk_limits(
            self.path_limits, self._limit_observations(result), self.risk.comparison_tolerance)
        return result

    def scenario(self) -> ShockScenario:
        """Factor shocks and stock returns of the current episode."""
        return ShockScenario(self.portfolio.tickers, self.simulator.model.factors,
                             self._factor_path.copy(), self._return_path.copy(),
                             {"anchor_date": self.anchor.date().isoformat(),
                              "surprisals": list(self._surprisals)})


def anchor_states(
    source: pd.DataFrame, columns: tuple[str, ...], start: str, end: str
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Complete market-state rows in ``[start, end]`` with their mean and std (for standardisation)."""
    missing = [c for c in columns if c not in source.columns]
    if missing:
        raise ValueError(f"state columns not in data: {missing}")
    states = select_window(source[list(columns)], start, end).dropna()
    return states, states.mean(), states.std(ddof=1)


def make_env(config: ProjectConfig, source: pd.DataFrame, model: ShockModel | None = None) -> StressTestEnv:
    """Build the environment from config and the Stage 2 dataset.

    The shock model (if not given) and the state standardisation are fitted on
    ``simulation.fit_window``; anchors come from ``simulation.anchor_window``.
    """
    sim = config.simulation
    splits = config.data.splits.as_dict()
    universe = [a.alias for a in config.data.enabled_assets("stock")]
    portfolio = Portfolio.from_config(config.portfolio, universe=universe)
    fit_start, fit_end = splits[sim.fit_window]
    if model is None:
        model = fit_shock_model(source, portfolio.tickers, sim, fit_start, fit_end,
                                config.portfolio.return_column_template)
    _, mean, sd = anchor_states(source, sim.state_columns, fit_start, fit_end)
    a_start, a_end = splits[sim.anchor_window]
    states, _, _ = anchor_states(source, sim.state_columns, a_start, a_end)
    return StressTestEnv(portfolio, ShockSimulator(model, sim), config.risk, config.environment,
                         sim, states, mean, sd)
