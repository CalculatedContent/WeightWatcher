import numpy as np

from weightwatcher.trap_histograms import _largest_trap_elements, _trap_color


def test_largest_trap_elements_single_peak():
    T = np.array([[0.1, -0.2], [0.4, -3.0]])
    vals = _largest_trap_elements(T)
    assert vals.shape == (1,)
    assert np.isclose(vals[0], -3.0)


def test_largest_trap_elements_multiple_peaks():
    T = np.array([[1.0, -2.5], [2.5, -0.1]])
    vals = np.sort(_largest_trap_elements(T))
    assert vals.shape == (2,)
    assert np.allclose(vals, np.array([-2.5, 2.5]))


def test_trap_color_varies_by_assessment():
    mixed = _trap_color(1, "mixed")
    risky = _trap_color(1, "localized_risky")
    benign = _trap_color(1, "benign_diffuse")

    assert len(mixed) == 3
    assert len(risky) == 3
    assert len(benign) == 3

    # risky should be darker than mixed; benign should be lighter than mixed.
    assert np.mean(risky) < np.mean(mixed)
    assert np.mean(benign) > np.mean(mixed)
