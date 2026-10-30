"""Calibrated plausibility of factor shocks under the fitted Student-t.

Per day: for factor shock ``f`` with Mahalanobis distance
``delta = (f - mu_f)' Sigma_f^-1 (f - mu_f)`` under the factor-only Student-t
``t_nu_f(mu_f, Sigma_f)`` fitted on training factor days (dimension ``k``),
``delta / k ~ F(k, nu_f)``. The tail probability

    p = P(F(k, nu_f) >= delta / k)

is the model probability of a factor day at least as extreme (in the
model's elliptical sense) as ``f``. It accounts for factor correlation (SPY
-5% with QQQ +5% is far less likely than both -5%) and fat tails. The
surprisal ``s = -ln p`` is the per-day implausibility; Mahalanobis distance is
reported only as a diagnostic, never as a probability.

Path: if model days were independent, ``p_t ~ U(0,1)`` and ``-ln p_t ~ Exp(1)``,
so ``S = sum_t s_t ~ Gamma(T, 1)`` and ``P_path = P(Gamma(T, 1) >= S)`` (Fisher's
method) is the probability that a random ``T``-day model path is at least this
extreme overall. Independence ignores volatility clustering, so P_path is a
model-relative score, not a real-world probability: on the training data,
3.75% of actual 5-day windows have P_path < 1%.

Empirical calibration: :func:`historical_path_percentile` ranks a path's
log P_path among all actual training windows of the same length (share of real
windows at least as extreme). This is what decides whether a breach is
plausible ("no more extreme than the most extreme ``min_plausibility`` share
of real weeks").
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from scipy import stats

from rlstress.shocks.factor_model import ShockModel
from rlstress.shocks.tdist import mahalanobis_sq


@dataclass(frozen=True)
class Plausibility:
    """Plausibility of one daily factor shock."""

    mahalanobis_sq: float
    f_statistic: float
    tail_probability: float
    surprisal: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return asdict(self)


def factor_plausibility(model: ShockModel, f: np.ndarray) -> Plausibility:
    """Tail probability and surprisal of one factor shock vector."""
    delta = float(mahalanobis_sq(np.asarray(f, dtype=float), model.plaus_mu, model.plaus_sigma))
    stat = delta / model.k
    log_sf = float(stats.f.logsf(stat, model.k, model.plaus_nu))
    return Plausibility(delta, stat, float(np.exp(log_sf)), -log_sf)


def path_log_tail_probability(surprisals: Sequence[float]) -> float:
    """``ln P(Gamma(T, 1) >= sum surprisals)`` (0.0 for an empty path); accurate for tiny probabilities."""
    if len(surprisals) == 0:
        return 0.0
    return float(stats.gamma.logsf(float(np.sum(surprisals)), a=len(surprisals)))


def historical_path_percentile(model: ShockModel, surprisals: Sequence[float]) -> float:
    """Share of actual training windows of the same length whose path is at least as extreme.

    1.0 for an empty path. Raises ``ValueError`` if the path is longer than the
    calibrated horizon.
    """
    t = len(surprisals)
    if t == 0:
        return 1.0
    key = str(t)
    if key not in model.path_calibration:
        raise ValueError(f"no path calibration for {t}-day windows")
    ref = np.asarray(model.path_calibration[key])
    return float(np.searchsorted(ref, path_log_tail_probability(surprisals), side="right") / len(ref))


def plausibility_log_threshold(model: ShockModel, t: int, min_plausibility: float) -> float:
    """Path log tail probability at which a ``t``-day path becomes plausible.

    ``historical_path_percentile(path) >= q``  <=>  ``path_log_tail_probability(path) >= L*_t`` with
    ``L*_t = ref[k - 1]``, ``ref`` the sorted training calibration and ``k = ceil(q * len(ref))``.
    """
    key = str(t)
    if key not in model.path_calibration:
        raise ValueError(f"no path calibration for {t}-day windows")
    if not 0.0 < min_plausibility < 1.0:
        raise ValueError(f"min_plausibility must be in (0, 1), got {min_plausibility!r}")
    ref = model.path_calibration[key]
    k = max(1, math.ceil(round(min_plausibility * len(ref), 9)))
    return float(ref[k - 1])


def implausibility_credit_fraction(log_tail: float, log_threshold: float) -> float:
    """Graded closeness to the plausibility threshold in [0, 1]: ``L* / ell`` for ``ell < L*``, else 1.

    Continuous and increasing in ``ell`` (more plausible -> more credit), equal to 1 at the threshold and
    tending to 0 for ever more extreme paths (0 when ``ell`` underflows to -inf).
    """
    if log_tail >= log_threshold:
        return 1.0
    if not np.isfinite(log_tail) or log_threshold >= 0.0:
        return 0.0
    return float(np.clip(log_threshold / log_tail, 0.0, 1.0))


def path_tail_probability(surprisals: Sequence[float]) -> float:
    """``P(Gamma(T, 1) >= sum surprisals)`` for ``T = len(surprisals)`` (1.0 for an empty path)."""
    if len(surprisals) == 0:
        return 1.0
    return float(stats.gamma.sf(float(np.sum(surprisals)), a=len(surprisals)))
