"""Regression coverage for the optional lambda_min ERG normalization."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from weightwatcher.RMT_Util import detX_constraint
from weightwatcher.constants import DEFAULT_PARAMS, ERG_NORMALIZATION
from weightwatcher.weightwatcher import FrameworkLayer, WeightWatcher, WWLayer


ERG_COLUMNS = [
    "detX_num", "num_ERG_spikes", "ERG_gap", "detX_val",
    "detX_val_unrescaled",
]


def _layer(evals, num_pl_spikes=None):
    framework_layer = FrameworkLayer(None, 0, "test_dense")
    layer = WWLayer(framework_layer, make_weights=False)
    layer.evals = None if evals is None else np.asarray(evals, dtype=float)
    if num_pl_spikes is not None:
        layer.add_column("num_pl_spikes", num_pl_spikes)
    return layer


def _params(**overrides):
    params = DEFAULT_PARAMS.copy()
    params.update(erg_normalization="lambda_min", plot=False, savefig=False)
    params.update(overrides)
    return params


@pytest.mark.parametrize("rescale", [False, True])
@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_analytic_tail_is_scale_invariant_and_does_not_mutate(rescale, scale):
    # The retained log ratios are [0, 1, 2], whose mean is exactly one.
    evals = scale * np.array([0.1, 1.0, np.e, np.e ** 2])
    original = evals.copy()
    with np.errstate(all="raise"):
        result = detX_constraint(evals, rescale=rescale, normalization="lambda_min")
    assert result == (3, 1)
    np.testing.assert_array_equal(evals, original)


def test_zero_prefix_preserves_original_cutoff_index():
    evals = [0.0, 0.0, 0.1, 1.0, np.e, np.e ** 2]
    assert detX_constraint(evals, normalization="lambda_min") == (3, 3)


def test_cutoff_moves_with_the_candidate_tail():
    # Adding smaller bulk eigenvalues must not change the normalization of
    # the same exact retained tail; the full-spectrum minimum is irrelevant.
    tail = np.exp([0.0, 1.0, 2.0])
    assert detX_constraint(tail, normalization="lambda_min") == (3, 0)
    evals = np.r_[1e-6, 0.01, 0.1, tail]
    assert detX_constraint(evals, normalization="lambda_min") == (3, 3)


def test_extreme_dynamic_range_does_not_form_an_overflowing_ratio():
    with np.errstate(all="raise"):
        assert detX_constraint([1e-300, 1e300], normalization="lambda_min") == (2, 0)


def test_cutoff_cannot_split_equal_eigenvalues():
    # Selecting just [1, e**2] would give an exact root, but would split the
    # two-dimensional eigenspace at lambda=1.
    assert detX_constraint([1.0, 1.0, np.e ** 2], normalization="lambda_min") == (3, 0)


def test_equal_residuals_prefer_the_larger_tail():
    # Both the full tail and [e**2, e**4] satisfy the condition exactly.
    assert detX_constraint(np.exp([1.5, 2.0, 4.0]), normalization="lambda_min") == (3, 0)


def test_finite_zipf_spectrum_selects_full_tail_without_a_zero_crossing():
    evals = 1.0 / np.arange(32, 0, -1, dtype=float)
    assert detX_constraint(evals, normalization="lambda_min") == (32, 0)


@pytest.mark.parametrize("evals", [
    None, [], [1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0],
    [-1.0, 2.0], [1.0, np.nan], [1.0, np.inf], [2.0, 1.0], [[1.0, 2.0]],
])
def test_invalid_or_degenerate_spectra_have_no_tail(evals):
    with np.errstate(all="raise"):
        assert detX_constraint(evals, normalization="lambda_min") == (0, 0)


@pytest.mark.parametrize("rescale", [False, True])
def test_default_preserves_the_legacy_zero_boundary(rescale):
    evals = np.r_[np.zeros(90), np.ones(10)]
    assert detX_constraint(evals, rescale=rescale) == (11, 89)
    assert detX_constraint(evals, rescale=rescale, normalization="frobenius") == (11, 89)
    assert detX_constraint(evals, rescale=rescale, normalization="lambda_min") == (0, 0)


def test_apply_detx_reports_expected_metrics_without_changing_the_esd():
    layer = _layer([0.1, 1.0, np.e, np.e ** 2], num_pl_spikes=2)
    original = layer.evals.copy()
    result = WeightWatcher().apply_detX(layer, params=_params())
    assert result is layer
    assert layer.columns == ["num_pl_spikes"] + ERG_COLUMNS
    assert layer.detX_num == layer.num_ERG_spikes == 3
    assert layer.ERG_gap == 1
    assert layer.detX_val == pytest.approx(1.0)
    assert layer.detX_val_unrescaled == pytest.approx(1.0 / np.e)
    np.testing.assert_array_equal(layer.evals, original)


@pytest.mark.parametrize("evals", [None, [], [0.0, 0.0], [1.0, 1.0]])
def test_apply_detx_no_tail_has_nan_cutoffs(evals):
    layer = _layer(evals)
    WeightWatcher().apply_detX(layer, params=_params(plot=True))
    assert layer.columns == ERG_COLUMNS
    assert layer.detX_num == layer.num_ERG_spikes == 0
    assert np.isnan(layer.detX_val)
    assert np.isnan(layer.detX_val_unrescaled)
    assert np.isnan(layer.ERG_gap)


def test_params_without_new_key_still_use_legacy_normalization():
    evals = np.r_[np.zeros(90), np.ones(10)]
    params = DEFAULT_PARAMS.copy()
    params.pop(ERG_NORMALIZATION)
    params.update(plot=False, savefig=False)
    layer = _layer(evals, num_pl_spikes=10)
    WeightWatcher().apply_detX(layer, params=params)
    assert layer.detX_num == layer.num_ERG_spikes == 11
    assert layer.ERG_gap == 1
    assert layer.detX_val == layer.detX_val_unrescaled == 0.0


@pytest.mark.parametrize("with_powerlaw", [False, True])
def test_plot_uses_selected_cutoff_with_optional_powerlaw_columns(monkeypatch, with_powerlaw):
    layer = _layer([0.1, 1.0, np.e, np.e ** 2])
    if with_powerlaw:
        layer.add_column("xmin", np.e)
        # A clipped fit need not extend to the maximum eigenvalue.
        layer.add_column("xmax", np.e)
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(plt, "clf", lambda: None)
    fig, ax = plt.subplots()
    try:
        WeightWatcher().apply_detX(layer, params=_params(plot=True))
        assert len(ax.lines) == (2 if with_powerlaw else 1)
        np.testing.assert_allclose(ax.lines[0].get_xdata(), -1.0 / np.log(10.0))
        if with_powerlaw:
            np.testing.assert_allclose(ax.lines[1].get_xdata(), 0.0, atol=1e-14)
        assert np.isnan(layer.ERG_gap)
    finally:
        plt.close(fig)


def test_invalid_normalization_is_rejected():
    with pytest.raises(ValueError, match="normalization"):
        detX_constraint([1.0, 2.0], normalization="unknown")
    params = _params(erg_normalization="unknown")
    assert not WeightWatcher.valid_params(params)
    with pytest.raises(ValueError, match="erg_normalization"):
        WeightWatcher().apply_detX(_layer([1.0, 2.0]), params=params)


def _torch_model(evals):
    torch = pytest.importorskip("torch")
    evals = np.asarray(evals, dtype=float)
    model = torch.nn.Sequential(torch.nn.Linear(len(evals), len(evals), bias=False)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(np.diag(np.sqrt(evals))))
    return model


def test_analyze_new_option_and_alias_preserve_other_metrics():
    evals = np.r_[np.exp(np.linspace(-9.0, -2.0, 9)), 1.0, np.e, np.e ** 2]
    watcher = WeightWatcher(_torch_model(evals))
    options = dict(plot=False, savefig=False, svd_method="accurate")
    default = watcher.analyze(ERG=True, **options)
    frobenius = watcher.analyze(ERG=True, erg_normalization="frobenius", **options)
    new_erg = watcher.analyze(ERG=True, erg_normalization="lambda_min", **options)
    new_detx = watcher.analyze(detX=True, erg_normalization="lambda_min", **options)
    pd.testing.assert_frame_equal(default, frobenius)
    pd.testing.assert_frame_equal(new_erg, new_detx)
    assert len(new_erg) == 1
    assert new_erg.detX_num.iloc[0] == new_erg.num_ERG_spikes.iloc[0] == 3
    assert new_erg.detX_val.iloc[0] == pytest.approx(1.0)
    assert new_erg.detX_val_unrescaled.iloc[0] == pytest.approx(1.0 / np.e)
    assert new_erg.ERG_gap.iloc[0] == 3 - new_erg.num_pl_spikes.iloc[0]
    pd.testing.assert_frame_equal(default.drop(columns=ERG_COLUMNS), new_erg.drop(columns=ERG_COLUMNS))

    plain = watcher.analyze(**options)
    mode_without_flag = watcher.analyze(erg_normalization="lambda_min", **options)
    pd.testing.assert_frame_equal(plain, mode_without_flag)
    assert not set(ERG_COLUMNS).intersection(mode_without_flag.columns)
    with pytest.raises(Exception, match="params not valid"):
        watcher.analyze(ERG=True, erg_normalization="unknown", **options)


@pytest.mark.parametrize("zero_count, expected_count", [(90, 11), (0, 0)])
def test_default_smoothing_still_uses_the_legacy_tail(monkeypatch, zero_count, expected_count):
    # Legacy detX admits the zero boundary and retains 11 components here;
    # lambda_min has no valid tail because all positive eigenvalues are equal.
    evals = np.r_[np.zeros(zero_count), np.ones(100 - zero_count)]
    model = _torch_model(evals)
    original = model[0].weight.detach().numpy().copy()
    watcher = WeightWatcher(model)
    details = watcher.analyze(ERG=True, plot=False, savefig=False, svd_method="accurate")
    assert details.detX_num.iloc[0] == expected_count
    calls = []
    smooth = watcher.smooth_W

    def record_smoothing(W, n_comp, svd_method):
        calls.append(n_comp)
        return smooth(W, n_comp, svd_method)

    monkeypatch.setattr(watcher, "smooth_W", record_smoothing)
    watcher.SVDSmoothing(svd_method="accurate")
    assert calls == ([expected_count] if expected_count else [])
    # PyTorchLayer converts weights to float32 during reconstruction.
    np.testing.assert_allclose(model[0].weight.detach().numpy(), original, atol=1e-6)
