import numpy as np

from weightwatcher.analyze_weights import _fit_side_models


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
