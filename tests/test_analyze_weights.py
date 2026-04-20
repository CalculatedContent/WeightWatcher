import numpy as np

from weightwatcher.analyze_weights import (
    _fit_side_models,
    compute_lognormal_self_averaging_stats,
)


def _best(rows):
    best = [r for r in rows if r.get("is_best_fit", False)]
    assert len(best) == 1
    return best[0]["distribution"]


def test_fit_power_law_right_side():
    rng = np.random.default_rng(7)
    samples = (1.0 + rng.pareto(a=3.0, size=12000)).astype(float)
    rows = _fit_side_models(samples, side_label="right", min_points=64)
    assert _best(rows) == "power_law"


def test_fit_exponential_right_side():
    rng = np.random.default_rng(8)
    samples = rng.exponential(scale=2.0, size=12000).astype(float)
    rows = _fit_side_models(samples, side_label="right", min_points=64)
    assert _best(rows) == "exponential"


def test_fit_lognormal_right_side():
    rng = np.random.default_rng(9)
    samples = rng.lognormal(mean=0.0, sigma=0.5, size=12000).astype(float)
    rows = _fit_side_models(samples, side_label="right", min_points=64)
    assert _best(rows) == "lognormal"


def test_fit_laplace_left_side():
    rng = np.random.default_rng(10)
    samples = rng.laplace(loc=-5.0, scale=0.4, size=12000).astype(float)
    rows = _fit_side_models(samples, side_label="left", min_points=64)
    assert _best(rows) == "laplace"


def _fixed_lognormal_samples(mu, sigma2, n, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    z = (z - z.mean()) / z.std(ddof=0)
    y = mu + np.sqrt(sigma2) * z
    return np.exp(y)


def test_lognormal_right_self_averaging_regime():
    n = 200
    sigma2 = 1.0
    vals = _fixed_lognormal_samples(mu=0.2, sigma2=sigma2, n=n, seed=101)
    out = compute_lognormal_self_averaging_stats(vals, "right", min_samples=10, tol=0.05, classified_as_lognormal=True)
    assert out["lognormal_right_detected"] is True
    assert out["lognormal_right_sa_regime"] == "self_averaging"
    assert out["lognormal_right_non_self_averaging"] is False
    assert np.isclose(out["lognormal_right_sigma2"], sigma2, atol=1e-10)
    assert out["lognormal_right_n"] == n
    assert np.isclose(out["lognormal_right_sa_ratio"], np.exp(sigma2) / n)


def test_lognormal_right_marginal_regime():
    n = 40
    sigma2 = np.log(n)
    vals = _fixed_lognormal_samples(mu=-0.1, sigma2=sigma2, n=n, seed=102)
    out = compute_lognormal_self_averaging_stats(vals, "right", min_samples=10, tol=0.05, classified_as_lognormal=True)
    assert out["lognormal_right_detected"] is True
    assert out["lognormal_right_sa_regime"] == "marginal"
    assert out["lognormal_right_non_self_averaging"] is False
    assert np.isclose(out["lognormal_right_nsa_margin"], 0.0, atol=1e-10)


def test_lognormal_right_non_self_averaging_regime():
    n = 40
    sigma2 = np.log(n) + 0.5
    vals = _fixed_lognormal_samples(mu=0.0, sigma2=sigma2, n=n, seed=103)
    out = compute_lognormal_self_averaging_stats(vals, "right", min_samples=10, tol=0.05, classified_as_lognormal=True)
    assert out["lognormal_right_detected"] is True
    assert out["lognormal_right_sa_regime"] == "non_self_averaging"
    assert out["lognormal_right_non_self_averaging"] is True
    assert out["lognormal_right_nsa_margin"] > 0.0


def test_lognormal_left_negative_case():
    n = 120
    sigma2 = 1.3
    vals = -_fixed_lognormal_samples(mu=0.3, sigma2=sigma2, n=n, seed=104)
    out = compute_lognormal_self_averaging_stats(vals, "left", min_samples=10, tol=0.05, classified_as_lognormal=True)
    assert out["lognormal_left_detected"] is True
    assert np.isclose(out["lognormal_left_sigma2"], sigma2, atol=1e-10)
    assert out["lognormal_left_n"] == n


def test_non_lognormal_control_is_undetermined():
    rng = np.random.default_rng(105)
    vals = rng.uniform(low=0.01, high=1.0, size=300)
    out = compute_lognormal_self_averaging_stats(vals, "right", min_samples=10, tol=0.05, classified_as_lognormal=False)
    assert out["lognormal_right_detected"] is False
    assert out["lognormal_right_sa_regime"] == "undetermined"
    assert np.isnan(out["lognormal_right_non_self_averaging"])
    assert np.isnan(out["lognormal_right_sigma2"])
    assert np.isnan(out["lognormal_right_n"])
    assert np.isnan(out["lognormal_right_sa_ratio"])
