"""Stage 4 shock model: Student-t tools, factor fit, train-only fitting, plausibility, simulator."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rlstress.shocks import (
    InvalidActionError,
    InvalidSimulationError,
    ShockModel,
    ShockSimulator,
    factor_plausibility,
    fit_shock_model,
    historical_path_percentile,
    path_tail_probability,
)
from rlstress.shocks.factor_model import variance_inflation
from rlstress.shocks.tdist import (
    conditional_mvt,
    fit_mvt,
    mvt_logpdf,
    sample_mvt,
    schafer_strimmer_intensity,
)

from conftest import STAGE4_A, STAGE4_B, STAGE4_FACTORS


def _fit(source, config):
    s, e = config.data.splits.train
    return fit_shock_model(source, config.portfolio.tickers, config.simulation, s, e)


# --- Student-t utilities --------------------------------------------------------------------

def test_logpdf_matches_scipy() -> None:
    mu = np.array([0.1, -0.2])
    sigma = np.array([[1.0, 0.3], [0.3, 2.0]])
    x = np.array([[0.0, 0.0], [1.5, -2.0]])
    np.testing.assert_allclose(mvt_logpdf(x, mu, sigma, 4.0),
                               stats.multivariate_t(mu, sigma, df=4.0).logpdf(x))


def test_em_recovers_parameters() -> None:
    rng = np.random.default_rng(0)
    sigma = np.array([[1.0, 0.6], [0.6, 1.0]])
    x = sample_mvt(rng, np.zeros(2), sigma, 5.0, 20000)
    fit = fit_mvt(x, [3.0, 5.0, 10.0, 30.0], shrinkage=0.0)
    assert fit.nu == 5.0
    np.testing.assert_allclose(fit.sigma, sigma, atol=0.05)
    assert fit.converged


def test_shrinkage_intensity_bounds() -> None:
    rng = np.random.default_rng(1)
    lam_indep = schafer_strimmer_intensity(rng.standard_normal((50, 5)))
    lam_corr = schafer_strimmer_intensity(rng.multivariate_normal(np.zeros(3), np.full((3, 3), 0.9) + 0.1 * np.eye(3), 2000))
    assert 0 <= lam_corr < lam_indep <= 1


def test_conditional_t_scale_grows_with_conditioning_extremeness() -> None:
    mu = np.zeros(3)
    sigma = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.0], [0.2, 0.0, 1.0]])
    m_small, s_small, nu_c = conditional_mvt(mu, sigma, 4.0, np.array([0]), np.array([0.1]))
    m_big, s_big, _ = conditional_mvt(mu, sigma, 4.0, np.array([0]), np.array([5.0]))
    assert nu_c == 5.0
    np.testing.assert_allclose(m_big, [2.5, 1.0])  # S21 S11^-1 x
    assert np.all(np.diag(s_big) > np.diag(s_small))  # tail dependence via the mixing variable


# --- Factor model -----------------------------------------------------------------------------

def test_translation_exact_on_noise_free_data(s4_source, s4_config) -> None:
    exact = s4_source.copy()
    f = exact[list(STAGE4_FACTORS)].to_numpy()
    rng = np.random.default_rng(5)
    tiny = rng.standard_normal((len(exact), 2)) * 1e-7  # near-zero idiosyncratic noise
    exact[["SYNA_return", "SYNB_return"]] = STAGE4_A + f @ STAGE4_B.T + tiny
    m = _fit(exact, s4_config)
    np.testing.assert_allclose(m.raw_loadings, STAGE4_B, atol=1e-4)
    np.testing.assert_allclose(m.raw_intercept, STAGE4_A, atol=1e-6)
    shock = np.array([-0.02, -0.03, -0.01, 0.01])
    np.testing.assert_allclose(m.expected_returns(shock), STAGE4_A + STAGE4_B @ shock, atol=1e-5)


def test_translation_accurate_along_realistic_factor_moves(s4_source, s4_config) -> None:
    # SPY and QQQ are 0.9 correlated: single raw loadings are weakly identified, but
    # the systematic return for realistic factor moves is estimated accurately.
    m = _fit(s4_source, s4_config)
    f = s4_source.loc["2015-01-02":"2018-12-31", list(STAGE4_FACTORS)].to_numpy()
    fitted = f @ m.raw_loadings.T
    truth = f @ STAGE4_B.T
    r2 = 1 - ((fitted - truth) ** 2).sum(0) / ((truth - truth.mean(0)) ** 2).sum(0)
    assert np.all(r2 > 0.98)
    assert m.raw_loadings[0, :2].sum() == pytest.approx(STAGE4_B[0, :2].sum(), abs=0.05)


def test_orthogonalisation_handles_multicollinearity(s4_source, s4_config) -> None:
    m = _fit(s4_source, s4_config)
    d = m.diagnostics
    assert d["vif_raw_factors"]["SPY_return"] > 5
    assert max(d["vif_orthogonalised_factors"].values()) < 2
    assert d["condition_number_orth_factor_cov"] < d["condition_number_raw_factor_cov"]
    g = m.orthogonalize(np.array([0.01, 0.02, 0.0, 0.0]))
    assert g[0] == pytest.approx(0.01)  # market factor unchanged


def test_variance_inflation_of_orthogonal_columns_is_one() -> None:
    x = np.random.default_rng(3).standard_normal((5000, 3))
    assert all(v == pytest.approx(1.0, abs=0.01) for v in variance_inflation(x))


def test_fit_uses_training_rows_only(s4_source, s4_config) -> None:
    base = _fit(s4_source, s4_config)
    corrupted = s4_source.copy()
    later = corrupted.index > pd.Timestamp("2018-12-31")
    corrupted.loc[later, ["SPY_return", "SYNA_return"]] = 0.5  # absurd post-training values
    corrupted.loc[later, "SYNB_return"] = np.nan
    other = _fit(corrupted, s4_config)
    assert base.fit_end <= "2018-12-31" and other.fit_end == base.fit_end
    np.testing.assert_array_equal(base.sigma, other.sigma)
    np.testing.assert_array_equal(base.raw_loadings, other.raw_loadings)
    assert base.path_calibration == other.path_calibration


def test_model_roundtrip(s4_source, s4_config) -> None:
    m = _fit(s4_source, s4_config)
    back = ShockModel.from_dict(m.to_dict())
    np.testing.assert_array_equal(back.sigma, m.sigma)
    assert back.tickers == m.tickers and back.nu == m.nu


# --- Plausibility ---------------------------------------------------------------------------

def test_implausible_combinations_score_lower(s4_source, s4_config) -> None:
    m = _fit(s4_source, s4_config)
    coherent = factor_plausibility(m, np.array([-0.04, -0.05, -0.05, -0.06]))
    incoherent = factor_plausibility(m, np.array([-0.04, 0.05, -0.05, -0.06]))  # SPY down, QQQ up
    calm = factor_plausibility(m, np.zeros(4))
    assert incoherent.tail_probability < coherent.tail_probability < calm.tail_probability
    assert calm.tail_probability > 0.99
    assert incoherent.surprisal > coherent.surprisal


def test_plausibility_decreases_with_magnitude(s4_source, s4_config) -> None:
    m = _fit(s4_source, s4_config)
    p = [factor_plausibility(m, np.full(4, -c)).tail_probability for c in (0.01, 0.03, 0.06, 0.1)]
    assert all(a > b for a, b in zip(p, p[1:]))


def test_path_probability_and_percentile(s4_source, s4_config) -> None:
    m = _fit(s4_source, s4_config)
    assert path_tail_probability([]) == 1.0 and historical_path_percentile(m, []) == 1.0
    assert path_tail_probability([0.0, 0.0]) == pytest.approx(1.0)
    mild, extreme = [0.5] * 5, [12.0] * 5
    assert historical_path_percentile(m, extreme) < historical_path_percentile(m, mild) <= 1.0
    assert historical_path_percentile(m, extreme) == 0.0
    with pytest.raises(ValueError):
        historical_path_percentile(m, [0.1] * 6)


# --- Simulator ------------------------------------------------------------------------------

def test_action_bounds_and_validation(s4_source, s4_config) -> None:
    m = _fit(s4_source, s4_config)
    sim = ShockSimulator(m, s4_config.simulation)
    cfg = s4_config.simulation
    np.testing.assert_allclose(sim.bounds, np.minimum(cfg.max_factor_sigma * m.factor_sd, cfg.max_abs_factor_return))
    np.testing.assert_allclose(sim.factor_shock(np.ones(4)), sim.bounds)
    for bad in (np.ones(3), np.array([0, 0, np.nan, 0]), np.array([0, 0, 1.01, 0]), np.array([np.inf, 0, 0, 0])):
        with pytest.raises(InvalidActionError):
            sim.factor_shock(bad)


def test_zero_action_mean_mode_returns_intercepts(s4_source, s4_config) -> None:
    m = _fit(s4_source, s4_config)
    step = ShockSimulator(m, s4_config.simulation).step(np.zeros(4), np.random.default_rng(0), mode="mean")
    np.testing.assert_allclose(step.factor_shock, 0.0)
    assert np.all(np.abs(step.stock_returns) < 1e-3)
    assert step.residual_draws == 0


def test_sector_shock_hits_exposed_stock(s4_source, s4_config) -> None:
    m = _fit(s4_source, s4_config)
    sim = ShockSimulator(m, s4_config.simulation)
    step = sim.step(np.array([0, 0, -0.4, 0]), np.random.default_rng(0), mode="mean")
    syna, synb = step.stock_returns
    assert synb < syna  # SYNB loads on XLF, SYNA does not
    assert synb < -0.03


def test_sampled_residuals_are_seed_deterministic(s4_source, s4_config) -> None:
    sim = ShockSimulator(_fit(s4_source, s4_config), s4_config.simulation)
    a = sim.step(np.full(4, -0.3), np.random.default_rng(7)).stock_returns
    b = sim.step(np.full(4, -0.3), np.random.default_rng(7)).stock_returns
    c = sim.step(np.full(4, -0.3), np.random.default_rng(8)).stock_returns
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_impossible_returns_rejected(s4_source, s4_config) -> None:
    m = _fit(s4_source, s4_config)
    tight = dataclasses.replace(s4_config.simulation, max_abs_stock_return=0.01)
    sim = ShockSimulator(m, tight)
    with pytest.raises(InvalidSimulationError, match="factor-implied"):
        sim.step(np.full(4, -1.0), np.random.default_rng(0))
    ok = ShockSimulator(m, s4_config.simulation)
    for seed in range(20):  # every accepted draw respects the bounds
        r = ok.step(np.full(4, -1.0), np.random.default_rng(seed)).stock_returns
        assert np.all(r > -1) and np.all(np.abs(r) <= s4_config.simulation.max_abs_stock_return)


def test_simulated_moments_track_training_data(s4_source, s4_config) -> None:
    from rlstress.shocks.factor_model import training_matrix

    m = _fit(s4_source, s4_config)
    _, sim_r = ShockSimulator(m, s4_config.simulation).sample_days(np.random.default_rng(0), 40000)
    _, emp, _ = training_matrix(s4_source, m.tickers, m.factors, "2015-01-01", "2018-12-31")
    np.testing.assert_allclose(sim_r.std(axis=0), emp.std().to_numpy(), rtol=0.1)


# --- Plausibility calibration audit (regression tests) --------------------------------------
# The daily score is p = P(F(k, nu) >= delta / k) under the factor-only Student-t. The path decision is the
# empirical rank of the summed surprisal among actual training windows of the same length; the Gamma(T, 1)
# "path tail probability" is only a monotone transform of that sum and is NOT a calibrated probability when
# daily scores are dependent (volatility clustering).

def _day_surprisals(m, factors: np.ndarray) -> np.ndarray:
    from rlstress.shocks.tdist import mahalanobis_sq

    return -stats.f.logsf(mahalanobis_sq(factors, m.plaus_mu, m.plaus_sigma) / m.k, m.k, m.plaus_nu)


def test_f_transform_matches_simulation_of_the_t_model() -> None:
    from rlstress.shocks.tdist import mahalanobis_sq

    rng = np.random.default_rng(7)
    k, nu = 4, 3.5
    mu = np.array([0.001, 0.0005, -0.0002, 0.0])
    a = rng.standard_normal((k, k)) * 0.01
    sigma = a @ a.T + np.eye(k) * 1e-5
    x = sample_mvt(rng, mu, sigma, nu, 200_000)
    stat = mahalanobis_sq(x, mu, sigma) / k
    p = stats.f.sf(stat, k, nu)
    for level in (0.05, 0.01, 0.001):
        se = np.sqrt(level * (1 - level) / len(p))
        assert abs((p < level).mean() - level) < 5 * se
    for q in (0.5, 0.9, 0.99):
        assert np.quantile(stat, q) == pytest.approx(stats.f.ppf(q, k, nu), rel=0.03)


def test_daily_tail_probability_bounds_and_orientation(s4_source, s4_config) -> None:
    m = _fit(s4_source, s4_config)
    rng = np.random.default_rng(3)
    for _ in range(200):
        direction = rng.standard_normal(4) * 0.02
        ps = [factor_plausibility(m, m.plaus_mu + c * direction) for c in (0.0, 0.5, 1.0, 2.0, 4.0)]
        assert all(np.isfinite(x.surprisal) and 0.0 <= x.tail_probability <= 1.0 for x in ps)
        assert all(x.tail_probability == pytest.approx(np.exp(-x.surprisal)) for x in ps)
        # moving further from the model centre along any ray never makes a day more plausible
        assert all(a.tail_probability >= b.tail_probability for a, b in zip(ps, ps[1:]))
    assert factor_plausibility(m, m.plaus_mu).tail_probability == pytest.approx(1.0)


def test_percentile_is_empirical_share_of_window_sums(s4_source, s4_config) -> None:
    from rlstress.shocks.factor_model import training_matrix

    m = _fit(s4_source, s4_config)
    f, _, _ = training_matrix(s4_source, m.tickers, m.factors, *s4_config.data.splits.train)
    day = _day_surprisals(m, f.to_numpy())
    rng = np.random.default_rng(11)
    for t in range(1, s4_config.simulation.horizon_days + 1):
        sums = np.convolve(day, np.ones(t), mode="valid")
        assert len(m.path_calibration[str(t)]) == len(sums)          # overlapping windows, one per start day
        for candidate in [list(rng.exponential(1.0, t)) for _ in range(50)] + [list(day[i:i + t]) for i in (0, 37, 400)]:
            expected = np.mean(sums >= np.sum(candidate) - 1e-9)     # share of real windows at least as extreme
            assert historical_path_percentile(m, candidate) == pytest.approx(expected, abs=1e-12)


def test_rank_orientation_ties_and_zero_shock(s4_source, s4_config) -> None:
    m = _fit(s4_source, s4_config)
    h = s4_config.simulation.horizon_days
    ranks = [historical_path_percentile(m, [c] * h) for c in (0.0, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0)]
    assert ranks[0] == 1.0                                            # zero shock: every real window is at least as extreme
    assert all(a >= b for a, b in zip(ranks, ranks[1:])) and ranks[-1] == 0.0
    ref = np.asarray(m.path_calibration[str(h)])
    # a candidate tied with the k-th most extreme real window counts that window (side="right"): rank >= k / N
    from rlstress.shocks.plausibility import path_log_tail_probability
    k_th = 10
    target = ref[k_th - 1]
    assert np.searchsorted(ref, target, side="right") >= k_th
    assert historical_path_percentile(m, []) == 1.0 and path_log_tail_probability([]) == 0.0


def test_scores_finite_over_feasible_actions_and_underflow_ranks_zero(s4_source, s4_config) -> None:
    import itertools

    from rlstress.shocks.plausibility import path_log_tail_probability

    m = _fit(s4_source, s4_config)
    sim = ShockSimulator(m, s4_config.simulation)
    h = s4_config.simulation.horizon_days
    worst = max(sim.step(np.array(c, dtype=float), np.random.default_rng(0), mode="mean").plausibility.surprisal
                for c in itertools.product([-1.0, 1.0], repeat=m.k))
    assert np.isfinite(path_log_tail_probability([worst] * h))        # no -inf can reach the RL observation
    # far beyond the action box Gamma logsf underflows to -inf; the rank must still be a clean 0.0
    assert path_log_tail_probability([1e6] * h) == -np.inf
    assert historical_path_percentile(m, [1e6] * h) == 0.0


def test_gamma_aggregation_needs_independence_but_empirical_rank_does_not() -> None:
    from types import SimpleNamespace

    from rlstress.shocks.plausibility import path_log_tail_probability

    rng = np.random.default_rng(5)
    t, n = 5, 120_000

    def clustered_uniforms(size: int) -> np.ndarray:
        # Gaussian-copula AR(1), phi = 0.9: every u_t is exactly U(0, 1), so only the serial dependence
        # (persistent calm and turbulent spells) differs from the Fisher/Gamma assumption
        from scipy.signal import lfilter

        e = rng.standard_normal(size) * np.sqrt(1 - 0.9 ** 2)
        e[0] = rng.standard_normal()
        return stats.norm.cdf(lfilter([1.0], [1.0, -0.9], e))

    def gamma_rate(u: np.ndarray) -> float:
        s = np.convolve(-np.log(u), np.ones(t), mode="valid")
        return float((stats.gamma.sf(s, a=t) < 0.01).mean())

    iid = rng.random(n)
    assert gamma_rate(iid) == pytest.approx(0.01, abs=0.002)          # Fisher/Gamma is exact under independence
    dep_ref, dep_new = clustered_uniforms(n), clustered_uniforms(n)
    # marginals are U(0, 1) by construction (Phi of a stationary N(0, 1)); an iid KS test would be invalid here
    assert abs(dep_new.mean() - 0.5) < 0.01 and abs((dep_new < 0.01).mean() - 0.01) < 0.003
    assert gamma_rate(dep_new) > 0.03                                  # badly anti-conservative under dependence alone

    s_ref = -np.log(dep_ref)
    calib = np.sort(stats.gamma.logsf(np.convolve(s_ref, np.ones(t), mode="valid"), a=t)).tolist()
    model = SimpleNamespace(path_calibration={str(t): calib})
    s_new = -np.log(dep_new)
    sums_new = np.convolve(s_new, np.ones(t), mode="valid")
    ranks = np.searchsorted(np.asarray(calib), stats.gamma.logsf(sums_new, a=t), side="right") / len(calib)
    for i in range(0, len(sums_new), 997):                             # the project function gives the same ranks
        assert historical_path_percentile(model, list(s_new[i:i + t])) == pytest.approx(ranks[i], abs=1e-12)
    assert (ranks < 0.01).mean() == pytest.approx(0.01, abs=0.004)    # empirical rank stays calibrated
    assert path_log_tail_probability(list(s_new[:t])) <= 0.0


def test_larger_synchronized_declines_are_not_more_plausible(s4_source, s4_config) -> None:
    m = _fit(s4_source, s4_config)
    sim = ShockSimulator(m, s4_config.simulation)
    h = s4_config.simulation.horizon_days
    rng = np.random.default_rng(0)
    losses, ranks = [], []
    for c in (0.05, 0.15, 0.3, 0.5, 0.8, 1.0):
        step = sim.step(np.full(m.k, -c), rng, mode="mean")
        losses.append(-float(np.prod(1 + np.repeat(step.stock_returns.mean(), h)) - 1))
        ranks.append(historical_path_percentile(m, [step.plausibility.surprisal] * h))
    assert all(a < b for a, b in zip(losses, losses[1:]))             # bigger shock, bigger loss
    assert all(a >= b for a, b in zip(ranks, ranks[1:]))              # but never more plausible
    assert ranks[-1] < s4_config.simulation.min_plausibility


def test_plausibility_fit_reproducible(s4_source, s4_config) -> None:
    a, b = _fit(s4_source, s4_config), _fit(s4_source, s4_config)
    np.testing.assert_array_equal(a.plaus_sigma, b.plaus_sigma)
    assert a.plaus_nu == b.plaus_nu and a.path_calibration == b.path_calibration
    sim_a, sim_b = ShockSimulator(a, s4_config.simulation), ShockSimulator(b, s4_config.simulation)
    x = sim_a.step(np.full(a.k, -0.2), np.random.default_rng(9))
    y = sim_b.step(np.full(b.k, -0.2), np.random.default_rng(9))
    np.testing.assert_array_equal(x.stock_returns, y.stock_returns)
    assert x.plausibility == y.plausibility
