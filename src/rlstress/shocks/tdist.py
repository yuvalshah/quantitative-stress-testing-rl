"""Multivariate Student-t utilities: EM fitting with shrinkage, density, conditioning, sampling.

Parametrisation: ``X ~ t_nu(mu, Sigma)`` with scatter ``Sigma``; for ``nu > 2``
the covariance is ``nu / (nu - 2) * Sigma``. Equivalently
``X = mu + Z / sqrt(W)``, ``Z ~ N(0, Sigma)``, ``W ~ chi2_nu / nu``. The shared
mixing variable ``W`` gives every component common tail behaviour: large moves
in one block (factors) come with wider dispersion in the other (residuals).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.special import gammaln


def mahalanobis_sq(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance of each row of ``x`` (or a single vector)."""
    chol = np.linalg.cholesky(sigma)
    diff = np.atleast_2d(x) - mu
    z = np.linalg.solve(chol, diff.T)
    out = np.sum(z * z, axis=0)
    return out if np.ndim(x) > 1 else out[0]


def mvt_logpdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, nu: float) -> np.ndarray:
    """Log density of the multivariate t for each row of ``x``."""
    d = len(mu)
    chol = np.linalg.cholesky(sigma)
    logdet = 2.0 * np.sum(np.log(np.diag(chol)))
    maha = mahalanobis_sq(np.atleast_2d(x), mu, sigma)
    return (gammaln((nu + d) / 2) - gammaln(nu / 2) - d / 2 * np.log(nu * np.pi)
            - 0.5 * logdet - (nu + d) / 2 * np.log1p(maha / nu))


def schafer_strimmer_intensity(x: np.ndarray) -> float:
    """Optimal shrinkage of the correlation matrix towards the identity (Schäfer & Strimmer 2005).

    ``lambda* = sum_{i!=j} Var(r_ij) / sum_{i!=j} r_ij^2``, clipped to [0, 1].
    Applied to the covariance this shrinks off-diagonal entries towards zero
    while keeping variances.
    """
    n = x.shape[0]
    z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    w = z[:, :, None] * z[:, None, :]
    w_bar = w.mean(axis=0)
    r = n / (n - 1) * w_bar
    var_r = n / (n - 1) ** 3 * ((w - w_bar) ** 2).sum(axis=0)
    off = ~np.eye(x.shape[1], dtype=bool)
    denom = float((r[off] ** 2).sum())
    if denom == 0:
        return 1.0
    return float(np.clip(var_r[off].sum() / denom, 0.0, 1.0))


def shrink_to_diagonal(s: np.ndarray, intensity: float) -> np.ndarray:
    """``(1 - lambda) S + lambda diag(S)``."""
    return (1.0 - intensity) * s + intensity * np.diag(np.diag(s))


@dataclass
class MVTFit:
    """Fitted multivariate t parameters for a fixed or selected ``nu``."""

    mu: np.ndarray
    sigma: np.ndarray
    nu: float
    loglik: float
    n_iter: int
    converged: bool
    shrinkage: float
    grid: list[dict[str, float]] = field(default_factory=list)

    @property
    def covariance(self) -> np.ndarray:
        """``nu / (nu - 2) * Sigma``."""
        return self.sigma * self.nu / (self.nu - 2.0)


def fit_mvt_em(
    x: np.ndarray, nu: float, shrinkage: float, max_iter: int = 1000, tol: float = 1e-9
) -> MVTFit:
    """EM estimate of ``mu`` and ``Sigma`` for fixed ``nu`` (Liu & Rubin 1995).

    E-step weights ``w_i = (nu + d) / (nu + delta_i)``; M-step
    ``mu = sum w_i x_i / sum w_i``, ``S = mean(w_i (x_i - mu)(x_i - mu)')`` followed by
    shrinkage of ``S`` towards its diagonal with the given intensity (regularised EM).
    """
    n, d = x.shape
    mu = x.mean(axis=0)
    sigma = shrink_to_diagonal(np.cov(x, rowvar=False, ddof=1) * (nu - 2) / nu, shrinkage)
    prev = -np.inf
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        delta = mahalanobis_sq(x, mu, sigma)
        w = (nu + d) / (nu + delta)
        mu = (w[:, None] * x).sum(axis=0) / w.sum()
        diff = x - mu
        s = (w[:, None] * diff).T @ diff / n
        sigma = shrink_to_diagonal(s, shrinkage)
        ll = float(mvt_logpdf(x, mu, sigma, nu).sum())
        if abs(ll - prev) <= tol * max(1.0, abs(ll)):
            converged = True
            break
        prev = ll
    return MVTFit(mu, sigma, float(nu), ll, it, converged, shrinkage)


def fit_mvt(
    x: np.ndarray, dof_grid: Sequence[float], shrinkage: float, max_iter: int = 1000, tol: float = 1e-9
) -> MVTFit:
    """Profile likelihood over ``dof_grid``: EM for each ``nu``, keep the best log-likelihood."""
    fits = [fit_mvt_em(x, nu, shrinkage, max_iter, tol) for nu in dof_grid]
    best = max(fits, key=lambda f: f.loglik)
    best.grid = [{"nu": f.nu, "loglik": f.loglik, "converged": f.converged} for f in fits]
    return best


def conditional_mvt(
    mu: np.ndarray, sigma: np.ndarray, nu: float, given: np.ndarray, x_given: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Distribution of the remaining components given ``x[given] = x_given``.

    For ``X = (X1, X2) ~ t_nu``: ``X2 | X1 = x1 ~ t_{nu + d1}(mu_2|1, c * Sigma_22|1)`` with
    ``mu_2|1 = mu_2 + S21 S11^-1 (x1 - mu_1)``, ``Sigma_22|1 = S22 - S21 S11^-1 S12`` and
    ``c = (nu + delta_1) / (nu + d1)``, ``delta_1`` the Mahalanobis distance of ``x1``.
    """
    d = len(mu)
    rest = np.setdiff1d(np.arange(d), given)
    s11 = sigma[np.ix_(given, given)]
    s21 = sigma[np.ix_(rest, given)]
    s22 = sigma[np.ix_(rest, rest)]
    diff = x_given - mu[given]
    solve = np.linalg.solve(s11, diff)
    mu_c = mu[rest] + s21 @ solve
    sigma_c = s22 - s21 @ np.linalg.solve(s11, s21.T)
    delta = float(diff @ solve)
    d1 = len(given)
    scale = (nu + delta) / (nu + d1)
    return mu_c, scale * sigma_c, nu + d1


def sample_mvt(
    rng: np.random.Generator, mu: np.ndarray, sigma: np.ndarray, nu: float, size: int
) -> np.ndarray:
    """``size`` draws of ``mu + Z / sqrt(W)``, ``Z ~ N(0, Sigma)``, ``W ~ chi2_nu / nu``."""
    z = rng.multivariate_normal(np.zeros(len(mu)), sigma, size=size, method="cholesky")
    w = rng.chisquare(nu, size=size) / nu
    return mu + z / np.sqrt(w)[:, None]
