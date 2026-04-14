import numpy as np

from weightwatcher.trap_histograms import (
    _format_hist_value_label,
    _largest_trap_elements,
    _largest_trap_weight_values,
    _significant_trap_components,
    _trap_color,
)


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


def test_largest_trap_weight_values_uses_weight_entries_at_peak_locations():
    T = np.array([[0.1, -0.2], [0.4, -3.0]])
    W = np.array([[0.1, -0.2], [0.4, -0.8]])
    vals = _largest_trap_weight_values(T, W)
    assert vals.shape == (1,)
    assert np.isclose(vals[0], -0.8)


def test_largest_trap_weight_values_falls_back_on_shape_mismatch():
    T = np.array([[1.0, -2.5], [2.5, -0.1]])
    W = np.array([1.0, 2.0, 3.0])
    vals = np.sort(_largest_trap_weight_values(T, W))
    assert vals.shape == (2,)
    assert np.allclose(vals, np.array([-2.5, 2.5]))


def test_rank1_component_entry_can_exceed_observed_weight_entry():
    # W is the sum of components; cancellation can keep W entries small even when
    # a single component has a larger-magnitude value at the same index.
    trap_component = np.array([[2.0, 0.0], [0.0, 0.0]])
    cancelling_component = np.array([[-1.7, 0.0], [0.0, 0.0]])
    W = trap_component + cancelling_component

    assert np.max(np.abs(trap_component)) > np.max(np.abs(W))


def test_significant_trap_components_filters_by_relative_threshold():
    T = np.array([[1.0, 0.2], [0.31, -0.05]])
    W = np.array([[10.0, 20.0], [30.0, 40.0]])
    x_vals, rel_vals, signed = _significant_trap_components(T, W, min_rel_coeff=0.3)

    assert np.allclose(np.sort(x_vals), np.array([10.0, 30.0]))
    assert np.all(rel_vals >= 0.3)
    assert signed.shape == x_vals.shape


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


def test_format_hist_value_label_compact_precision():
    assert _format_hist_value_label(0.123456) == "0.1235"
    assert _format_hist_value_label(12.3456) == "12.35"
