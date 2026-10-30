"""Hybrid factor + residual shock model, fitted on the training split only.

Model (daily simple returns):

1. Factors ``f`` = configured indicator returns (default SPY, QQQ, XLF, XLE).
   QQQ, XLF and XLE are highly correlated with SPY, so non-market factors are
   orthogonalised against the market factor with OLS on training data:
   ``g_M = f_M``, ``g_j = f_j - c_j - h_j f_M``. Loadings are estimated on ``g``
   (well conditioned) and mapped back to raw factors, ``g = A f - c``.
2. Stocks: ``r_i = alpha_i + beta_i' g + e_i`` (OLS on ``g``), so
   ``r = a + B f + e`` with ``a = alpha + beta c``, ``B = beta A``.
3. Joint distribution: ``z = (f, e)`` (factors and stock residuals) follows a
   multivariate Student-t fitted by EM with shrinkage of the scatter towards its
   diagonal. The common ``nu`` and shared mixing variable keep tail dependence
   between factors and residuals; the full residual scatter keeps residual
   correlation between stocks. Used to simulate residuals given factor shocks.
4. Plausibility model: a separate Student-t fitted to the factors alone
   (``plaus_*``). The joint ``nu`` is driven mostly by the 8 residual
   dimensions; the factor-only fit is better calibrated for scoring factor
   days (see diagnostics), which is the only thing it is used for.
5. Path calibration: for every window length ``t = 1..horizon`` the sorted log
   Fisher path tail probabilities of all actual ``t``-day training windows
   (``path_calibration``). Real returns cluster in volatility, so the i.i.d.
   Fisher probability alone would call real crash weeks implausible; ranking a
   simulated path against real windows corrects for that.

Rows with any missing factor or stock return are excluded and counted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from rlstress.config import SimulationConfig
from rlstress.exceptions import InsufficientDataError, ReturnDataError
from rlstress.portfolio import select_window, validate_return_frame
from rlstress.shocks.tdist import fit_mvt, mahalanobis_sq, schafer_strimmer_intensity


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS with intercept: returns (coefficients incl. intercept first, residuals)."""
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef, y - design @ coef


def variance_inflation(x: np.ndarray) -> list[float]:
    """VIF of each column: ``1 / (1 - R^2)`` from regressing it on the others."""
    out = []
    for j in range(x.shape[1]):
        others = np.delete(x, j, axis=1)
        _, resid = _ols(x[:, j], others)
        r2 = 1 - resid.var() / x[:, j].var()
        out.append(float(1 / (1 - r2)) if r2 < 1 else float("inf"))
    return out


@dataclass
class ShockModel:
    """Fitted factor model and joint Student-t distribution of (factors, residuals)."""

    tickers: tuple[str, ...]
    factors: tuple[str, ...]
    market_factor: str
    orth_intercept: np.ndarray       # c_j (0 for the market factor)
    orth_slope: np.ndarray           # h_j (0 for the market factor)
    alpha: np.ndarray                # intercepts on orthogonalised factors (n,)
    beta: np.ndarray                 # loadings on orthogonalised factors (n, k)
    raw_intercept: np.ndarray        # a (n,)
    raw_loadings: np.ndarray         # B (n, k)
    mu: np.ndarray                   # t location of (f, e)  (k + n,)
    sigma: np.ndarray                # t scatter of (f, e)   (k + n, k + n)
    nu: float
    shrinkage: float
    plaus_mu: np.ndarray             # factor-only t location (k,)
    plaus_sigma: np.ndarray          # factor-only t scatter (k, k)
    plaus_nu: float
    plaus_shrinkage: float
    path_calibration: dict[str, list[float]]   # window length -> sorted log path tail probs
    fit_start: str
    fit_end: str
    n_obs: int
    n_excluded: int
    loglik: float
    dof_grid: list[dict[str, float]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def k(self) -> int:
        """Number of factors."""
        return len(self.factors)

    @property
    def factor_mu(self) -> np.ndarray:
        """Location of the factor block."""
        return self.mu[: self.k]

    @property
    def factor_sigma(self) -> np.ndarray:
        """Scatter of the factor block (the marginal of a multivariate t is t with the same nu)."""
        return self.sigma[: self.k, : self.k]

    @property
    def factor_sd(self) -> np.ndarray:
        """Model standard deviation of each factor, ``sqrt(nu/(nu-2) Sigma_jj)``."""
        return np.sqrt(np.diag(self.factor_sigma) * self.nu / (self.nu - 2.0))

    def expected_returns(self, f: np.ndarray) -> np.ndarray:
        """Factor-implied stock returns ``a + B f`` (residual excluded)."""
        return self.raw_intercept + self.raw_loadings @ np.asarray(f, dtype=float)

    def orthogonalize(self, f: np.ndarray) -> np.ndarray:
        """``g = A f - c`` for one factor vector."""
        f = np.asarray(f, dtype=float)
        m = f[self.factors.index(self.market_factor)]
        return f - self.orth_intercept - self.orth_slope * m

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable parameters."""
        out = asdict(self)
        for key, value in out.items():
            if isinstance(value, np.ndarray):
                out[key] = value.tolist()
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShockModel:
        """Inverse of :meth:`to_dict`."""
        arrays = ("orth_intercept", "orth_slope", "alpha", "beta", "raw_intercept",
                  "raw_loadings", "mu", "sigma", "plaus_mu", "plaus_sigma")
        kwargs = {k: (np.asarray(v, dtype=float) if k in arrays else v) for k, v in data.items()}
        kwargs["tickers"] = tuple(kwargs["tickers"])
        kwargs["factors"] = tuple(kwargs["factors"])
        return cls(**kwargs)


def training_matrix(
    source: pd.DataFrame,
    tickers: Sequence[str],
    factors: Sequence[str],
    start: str,
    end: str,
    return_template: str = "{ticker}_return",
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Complete-case factor and stock returns inside ``[start, end]`` only.

    Returns:
        Factor frame, stock frame (ticker columns), number of excluded rows.
    """
    stock_cols = [return_template.format(ticker=t) for t in tickers]
    validate_return_frame(source, [*factors, *stock_cols])
    window = select_window(source[[*factors, *stock_cols]], start, end)
    complete = window.dropna()
    if complete.index.max() > pd.Timestamp(end) or complete.index.min() < pd.Timestamp(start):
        raise ReturnDataError("training matrix leaked outside the fit window")
    stocks = complete[stock_cols].copy()
    stocks.columns = list(tickers)
    return complete[list(factors)], stocks, int(len(window) - len(complete))


def fit_shock_model(
    source: pd.DataFrame,
    tickers: Sequence[str],
    config: SimulationConfig,
    start: str,
    end: str,
    return_template: str = "{ticker}_return",
) -> ShockModel:
    """Fit the factor model and the joint Student-t on ``[start, end]`` (the training split).

    Raises:
        InsufficientDataError: If fewer than ``10 * (k + n)`` complete rows are available.
    """
    factors = list(config.factors)
    f_df, r_df, n_excluded = training_matrix(source, tickers, factors, start, end, return_template)
    k, n = len(factors), len(tickers)
    if len(f_df) < 10 * (k + n):
        raise InsufficientDataError(f"need at least {10 * (k + n)} complete rows, got {len(f_df)}")
    f = f_df.to_numpy(dtype=float)
    r = r_df.to_numpy(dtype=float)
    m_idx = factors.index(config.market_factor)

    orth_c = np.zeros(k)
    orth_h = np.zeros(k)
    g = f.copy()
    for j in range(k):
        if j == m_idx:
            continue
        coef, resid = _ols(f[:, j], f[:, [m_idx]])
        orth_c[j], orth_h[j] = coef
        g[:, j] = resid
    coef, resid = _ols(r, g)
    alpha, beta = coef[0], coef[1:].T          # (n,), (n, k)
    a_mat = np.eye(k)
    a_mat[:, m_idx] -= orth_h                  # g = A f - c
    raw_loadings = beta @ a_mat
    raw_intercept = alpha - beta @ orth_c

    z = np.column_stack([f, resid])
    shrinkage = (schafer_strimmer_intensity(z) if config.shrinkage_method == "schafer_strimmer"
                 else config.fixed_shrinkage)
    fit = fit_mvt(z, config.dof_grid, shrinkage, config.em_max_iter, config.em_tol)
    f_shrink = (schafer_strimmer_intensity(f) if config.shrinkage_method == "schafer_strimmer"
                else config.fixed_shrinkage)
    pfit = fit_mvt(f, config.dof_grid, f_shrink, config.em_max_iter, config.em_tol)
    day_surprisal = -stats.f.logsf(mahalanobis_sq(f, pfit.mu, pfit.sigma) / k, k, pfit.nu)
    path_calibration = {
        str(t): np.sort(stats.gamma.logsf(
            np.convolve(day_surprisal, np.ones(t), mode="valid"), a=t)).tolist()
        for t in range(1, config.horizon_days + 1)
    }

    r2 = 1 - resid.var(axis=0) / r.var(axis=0)
    resid_corr = np.corrcoef(resid, rowvar=False)
    off = ~np.eye(n, dtype=bool)
    lag1 = [float(np.corrcoef(resid[1:, i], resid[:-1, i])[0, 1]) for i in range(n)]
    kurt = (((resid - resid.mean(0)) ** 4).mean(0) / resid.var(0) ** 2 - 3.0)
    factor_corr = np.corrcoef(f, rowvar=False)
    diagnostics = {
        "r_squared": dict(zip(tickers, r2.round(6).tolist())),
        "residual_sd_daily": dict(zip(tickers, resid.std(axis=0, ddof=1).round(8).tolist())),
        "residual_excess_kurtosis": dict(zip(tickers, kurt.round(4).tolist())),
        "residual_lag1_autocorr": dict(zip(tickers, np.round(lag1, 4).tolist())),
        "residual_corr_max_abs": float(np.abs(resid_corr[off]).max()),
        "residual_corr_mean_abs": float(np.abs(resid_corr[off]).mean()),
        "residual_correlation": pd.DataFrame(resid_corr, index=tickers, columns=tickers).round(4).to_dict(),
        "factor_correlation_raw": pd.DataFrame(factor_corr, index=factors, columns=factors).round(4).to_dict(),
        "vif_raw_factors": dict(zip(factors, np.round(variance_inflation(f), 3).tolist())),
        "vif_orthogonalised_factors": dict(zip(factors, np.round(variance_inflation(g), 3).tolist())),
        "condition_number_raw_factor_cov": float(np.linalg.cond(np.cov(f, rowvar=False))),
        "condition_number_orth_factor_cov": float(np.linalg.cond(np.cov(g, rowvar=False))),
        "condition_number_t_scatter": float(np.linalg.cond(fit.sigma)),
        "em_converged": fit.converged,
        "em_iterations": fit.n_iter,
        "plausibility_model_converged": pfit.converged,
        "plausibility_model_dof_grid": pfit.grid,
        "max_abs_training_move_in_model_sd": dict(zip(factors, (np.abs(f).max(axis=0) / np.sqrt(
            np.diag(fit.sigma[:k, :k]) * fit.nu / (fit.nu - 2))).round(3).tolist())),
    }
    return ShockModel(
        tickers=tuple(tickers), factors=tuple(factors), market_factor=config.market_factor,
        orth_intercept=orth_c, orth_slope=orth_h, alpha=alpha, beta=beta,
        raw_intercept=raw_intercept, raw_loadings=raw_loadings,
        mu=fit.mu, sigma=fit.sigma, nu=fit.nu, shrinkage=shrinkage,
        plaus_mu=pfit.mu, plaus_sigma=pfit.sigma, plaus_nu=pfit.nu, plaus_shrinkage=f_shrink,
        path_calibration=path_calibration,
        fit_start=f_df.index.min().date().isoformat(), fit_end=f_df.index.max().date().isoformat(),
        n_obs=int(len(f_df)), n_excluded=n_excluded, loglik=fit.loglik, dof_grid=fit.grid,
        diagnostics=diagnostics,
    )
