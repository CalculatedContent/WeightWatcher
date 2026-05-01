import unittest
import numpy as np
import pandas as pd
import pytest

import weightwatcher as ww
from weightwatcher.weightwatcher import WeightWatcher

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class OneLayerTrapNet(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, W):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        out_dim, in_dim = W.shape
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor(W, dtype=torch.float32))

    def forward(self, x):
        return self.fc(x)


def _get_first_linear_weight_np(model):
    for m in model.modules():
        if TORCH_AVAILABLE and isinstance(m, nn.Linear):
            return m.weight.detach().cpu().numpy().copy()
    raise AssertionError("No Linear layer found")


def _make_one_layer_trap_model(seed=123, out_dim=96, in_dim=96, spike_strength=100.0):
    rng = np.random.RandomState(seed)
    W_noise = 0.005 * rng.standard_normal((out_dim, in_dim))
    u = np.zeros(out_dim)
    v = np.zeros(in_dim)
    u[3] = 1.0
    u[4] = 0.5
    v[7] = 1.0
    v[8] = -0.5
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    W = W_noise + spike_strength * np.outer(u, v)
    return OneLayerTrapNet(W), W


def _select_trap_row(trap_df):
    col = "B_absDelta_ipr_ovlamvar"
    if col in trap_df.columns:
        vals = pd.to_numeric(trap_df[col], errors="coerce")
        if np.isfinite(vals).any():
            return int(vals.idxmax())
    return int(trap_df.index[0])


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required")
def test_randomized_model_analyze_then_remove_full_workflow():
    model, W_orig = _make_one_layer_trap_model()
    watcher = ww.WeightWatcher(model=model)

    randomized_model, trap_state = watcher.randomize_model(model=model, layers=[], rng=123, return_state=True)
    assert randomized_model is not None
    assert isinstance(trap_state, dict)
    assert "permuted_ids" in trap_state
    assert "layers" in trap_state

    W_rand_before = _get_first_linear_weight_np(randomized_model)
    assert not np.allclose(W_rand_before, W_orig)

    trap_df, trap_state = watcher.analyze_traps(
        randomized_model=randomized_model,
        trap_state=trap_state,
        return_artifacts=True,
        trap_burden=True,
        plot=False,
        savefig=False,
    )

    W_rand_after_analyze = _get_first_linear_weight_np(randomized_model)
    assert np.allclose(W_rand_before, W_rand_after_analyze)
    assert isinstance(trap_df, pd.DataFrame)
    assert len(trap_df) > 0
    for col in ["layer_id", "trap_index", "B_absDelta_ipr_ovlamvar", "spectral_excess_abs", "ipr_lift_excess_pos", "ov_lam_weighted_var"]:
        assert col in trap_df.columns

    assert isinstance(trap_state, dict)
    assert "layers" in trap_state and len(trap_state["layers"]) > 0

    idx = _select_trap_row(trap_df)
    row = trap_df.loc[idx]
    single_trap_df = trap_df.loc[[idx]].copy()

    selected_layer_id = int(row["layer_id"])
    selected_trap_index = int(row["trap_index"])
    layer_state = trap_state["layers"][selected_layer_id]
    for key in ["artifacts", "U_perm", "S_perm", "Vh_perm", "permuted_ids", "permute_fingerprint"]:
        assert key in layer_state
    assert len(layer_state["artifacts"]) > 0

    artifact = None
    for a in layer_state["artifacts"]:
        if int(a.get("trap_index", -1)) == selected_trap_index:
            artifact = a
            break
    assert artifact is not None
    if "trap_mode_index" in row and "trap_mode_index" in artifact:
        assert int(row["trap_mode_index"]) == int(artifact["trap_mode_index"])
    if "sigma_perm" in row and "sigma_perm" in artifact:
        assert np.isclose(float(row["sigma_perm"]), float(artifact["sigma_perm"]), rtol=1e-5, atol=1e-8)

    W_before_remove = _get_first_linear_weight_np(randomized_model)
    ablated_model = watcher.remove_traps(
        randomized_model=randomized_model,
        traps=single_trap_df,
        trap_state=trap_state,
        plot=False,
    )
    assert ablated_model is not None
    W_after_remove = _get_first_linear_weight_np(randomized_model)
    assert W_after_remove.shape == W_before_remove.shape
    assert not np.allclose(W_after_remove, W_before_remove)

    with pytest.raises(ValueError, match="requires randomized_model"):
        watcher.remove_traps(model=model, traps=single_trap_df, trap_state=trap_state, plot=False)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required")
def test_randomized_workflow_remove_traps_uses_cached_artifacts_no_svd(monkeypatch):
    model, _ = _make_one_layer_trap_model()
    watcher = WeightWatcher(model=model)
    randomized_model, trap_state = watcher.randomize_model(model=model, rng=123, return_state=True)
    trap_df, trap_state = watcher.analyze_traps(
        randomized_model=randomized_model,
        trap_state=trap_state,
        return_artifacts=True,
        trap_burden=True,
        plot=False,
        savefig=False,
    )
    assert len(trap_df) > 0
    single_trap_df = trap_df.iloc[[0]].copy()

    from weightwatcher import remove_traps as remove_traps_ops

    def fail_collect(*args, **kwargs):
        raise AssertionError("remove_traps should use cached trap_state artifacts, not recollect artifacts")

    monkeypatch.setattr(remove_traps_ops, "collect_trap_artifacts", fail_collect)

    out = watcher.remove_traps(randomized_model=randomized_model, traps=single_trap_df, trap_state=trap_state, plot=False)
    assert out is not None


def test_randomized_model_workflow_public_api(monkeypatch):
    from tests.test_remove_traps import _single_trap_setup, make_ww_layer
    W, _, _, _ = _single_trap_setup(seed=111)
    ww_layer = make_ww_layer(W)
    watcher = WeightWatcher(model={"dummy_weight": np.array([1.0])})

    monkeypatch.setattr(
        watcher,
        "make_layer_iterator",
        lambda model=None, layers=None, params=None, base_model=None: [ww_layer],
    )

    randomized_model, trap_state = watcher.randomize_model(model={"dummy_weight": np.array([1.0])}, return_state=True, rng=123)
    assert isinstance(trap_state, dict)
    assert "permuted_ids" in trap_state

    trap_df, trap_state = watcher.analyze_traps(
        randomized_model=randomized_model,
        trap_state=trap_state,
        return_artifacts=True,
        trap_burden=True,
        plot=False,
        savefig=False,
    )
    assert isinstance(trap_df, pd.DataFrame)
    assert "B_absDelta_ipr_ovlamvar" in trap_df.columns
    assert "layers" in trap_state


def test_remove_traps_requires_randomized_model_when_trap_state():
    watcher = WeightWatcher(model={"dummy_weight": np.array([1.0])})
    with pytest.raises(ValueError, match="requires randomized_model"):
        watcher.remove_traps(model={"dummy_weight": np.array([1.0])}, trap_state={"layers": {}}, traps=pd.DataFrame([{"trap_index": 1}]), plot=False)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required")
def test_analyze_traps_return_artifacts_does_not_recollect_artifacts(monkeypatch):
    model, _ = _make_one_layer_trap_model()
    watcher = WeightWatcher(model=model)
    randomized_model, trap_state = watcher.randomize_model(model=model, rng=123, return_state=True)

    from weightwatcher import remove_traps as remove_traps_ops

    def fail_collect(*args, **kwargs):
        raise AssertionError("analyze_traps(return_artifacts=True) must not recollect artifacts or recompute SVD")

    monkeypatch.setattr(remove_traps_ops, "collect_trap_artifacts", fail_collect)

    trap_df, trap_state = watcher.analyze_traps(
        randomized_model=randomized_model,
        trap_state=trap_state,
        return_artifacts=True,
        trap_burden=True,
        plot=False,
        savefig=False,
    )
    assert len(trap_df) > 0
    assert "layers" in trap_state and len(trap_state["layers"]) > 0
