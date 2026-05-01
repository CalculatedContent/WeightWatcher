import inspect
import pytest
import numpy as np
torch = pytest.importorskip("torch")
import weightwatcher as ww
from weightwatcher.RMT_Util import permute_matrix, unpermute_matrix

class OneLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(32, 32, bias=False)
        with torch.no_grad():
            W = 0.02 * torch.randn(32, 32)
            u = torch.zeros(32); u[:3] = 1.0
            v = torch.zeros(32); v[:2] = 1.0
            W += 5.0 * torch.outer(u, v)
            self.fc.weight.copy_(W)
    def forward(self, x):
        return self.fc(x)

def test_randomized_model_analyze_then_remove_full_workflow_fast_mode():
    model = OneLayer()
    watcher = ww.WeightWatcher(model=model)
    randomized_model, trap_state = watcher.randomize_model(model=model, rng=123, return_state=True)
    trap_df, trap_state = watcher.analyze_traps(randomized_model=randomized_model, trap_state=trap_state, return_artifacts=True, trap_burden=True, trap_burden_mode="fast", bulk_mode_sample=10, plot=False)
    assert len(trap_df) > 0
    assert "B_absDelta_ipr_ovlamvar" in trap_df.columns
    assert "layers" in trap_state and trap_state["layers"]
    lid = int(trap_df.iloc[0]["layer_id"])
    ls = trap_state["layers"][lid]
    assert all(k in ls for k in ["U_perm", "S_perm", "Vh_perm", "artifacts"])
    assert int(ls["artifacts"][0]["trap_index"]) == int(trap_df.iloc[0]["trap_index"])
    old = randomized_model.fc.weight.detach().clone()
    ablated_model = watcher.remove_traps(randomized_model=randomized_model, traps=trap_df.iloc[[0]], trap_state=trap_state, plot=False)
    assert not torch.allclose(old, ablated_model.fc.weight.detach())
    with pytest.raises(ValueError):
        watcher.remove_traps(traps=trap_df.iloc[[0]], trap_state=trap_state, plot=False)

def test_public_api_signatures():
    rz = inspect.signature(ww.WeightWatcher.randomize_model)
    for name in ["model","layers","rng","return_state","pool","start_ids","svd_method","base_model","peft"]:
        assert name in rz.parameters
    a = inspect.signature(ww.WeightWatcher.analyze_traps)
    for name in ["randomized_model","trap_state","return_artifacts","trap_burden_mode","bulk_mode_sample","compute_original_basis","compute_full_bulk_reference","compute_original_trap_svd"]:
        assert name in a.parameters
    assert "already_randomized" not in a.parameters
    r = inspect.signature(ww.WeightWatcher.remove_traps)
    for name in ["randomized_model","trap_state","trap_artifacts"]:
        assert name in r.parameters
    assert "already_randomized" not in r.parameters

def test_fast_mode_skips_original_basis(monkeypatch):
    model = OneLayer()
    watcher = ww.WeightWatcher(model=model)
    randomized_model, trap_state = watcher.randomize_model(model=model, rng=123, return_state=True)
    monkeypatch.setattr(watcher, "compute_original_basis_for_traps", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("should not be called")))
    trap_df = watcher.analyze_traps(
        randomized_model=randomized_model, trap_state=trap_state,
        trap_burden=True, trap_burden_mode="fast", plot=False
    )
    assert len(trap_df) > 0

def test_fast_mode_skips_full_bulk_reference(monkeypatch):
    model = OneLayer()
    watcher = ww.WeightWatcher(model=model)
    randomized_model, trap_state = watcher.randomize_model(model=model, rng=123, return_state=True)
    monkeypatch.setattr(watcher, "compute_bulk_trap_reference_metrics", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("should not be called")))
    trap_df = watcher.analyze_traps(
        randomized_model=randomized_model, trap_state=trap_state,
        trap_burden=True, trap_burden_mode="fast", plot=False
    )
    assert len(trap_df) > 0

def test_analyze_traps_does_not_recollect_artifacts(monkeypatch):
    model = OneLayer()
    watcher = ww.WeightWatcher(model=model)
    randomized_model, trap_state = watcher.randomize_model(model=model, rng=123, return_state=True)
    import weightwatcher.remove_traps as rt
    monkeypatch.setattr(rt, "collect_trap_artifacts", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("should not be called")))
    trap_df, out_state = watcher.analyze_traps(
        randomized_model=randomized_model, trap_state=trap_state, return_artifacts=True,
        trap_burden=True, trap_burden_mode="fast", plot=False
    )
    assert len(trap_df) > 0
    assert "layers" in out_state

def test_remove_traps_uses_cached_artifacts(monkeypatch):
    model = OneLayer()
    watcher = ww.WeightWatcher(model=model)
    randomized_model, trap_state = watcher.randomize_model(model=model, rng=123, return_state=True)
    trap_df, trap_state = watcher.analyze_traps(
        randomized_model=randomized_model, trap_state=trap_state, return_artifacts=True,
        trap_burden=True, trap_burden_mode="fast", plot=False
    )
    import weightwatcher.remove_traps as rt
    monkeypatch.setattr(rt, "collect_trap_artifacts", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("should not be called")))
    ablated_model = watcher.remove_traps(randomized_model=randomized_model, traps=trap_df.iloc[[0]], trap_state=trap_state, plot=False)
    assert isinstance(ablated_model, OneLayer)

def test_randomize_model_stores_permuted_ids_int_keys_pool_false():
    model = OneLayer()
    watcher = ww.WeightWatcher(model=model)
    randomized_model, trap_state = watcher.randomize_model(model=model, rng=123, return_state=True, pool=False)
    assert randomized_model is not None
    assert trap_state["permuted_ids"]
    assert all(isinstance(k, int) for k in trap_state["permuted_ids"].keys())
    for lid in sorted(trap_state["permuted_ids"].keys()):
        assert trap_state["layers"][lid]["permuted_ids"] is not None

def test_cached_analyze_requires_randomized_model():
    model = OneLayer()
    watcher = ww.WeightWatcher(model=model)
    _, trap_state = watcher.randomize_model(model=model, rng=123, return_state=True, pool=False)
    with pytest.raises(ValueError, match="cached trap artifact analysis requires randomized_model"):
        watcher.analyze_traps(model=model, trap_state=trap_state, return_artifacts=True, pool=False)

def test_cached_analyze_missing_permuted_id_errors():
    model = OneLayer()
    watcher = ww.WeightWatcher(model=model)
    randomized_model, trap_state = watcher.randomize_model(model=model, rng=123, return_state=True, pool=False)
    randomized_layers = sorted(trap_state["permuted_ids"].keys())
    with pytest.raises(ValueError, match="Missing permute_ids for already-randomized layer_id"):
        watcher.analyze_traps(
            randomized_model=randomized_model,
            layers=randomized_layers + [999],
            trap_state=trap_state,
            permuted_ids=trap_state["permuted_ids"],
            return_artifacts=True,
            pool=False,
            plot=False,
        )

def test_rmt_util_unpermutes_randomized_layer_weight():
    model = OneLayer()
    watcher = ww.WeightWatcher(model=model)
    original = model.fc.weight.detach().cpu().numpy().copy()
    randomized_model, trap_state = watcher.randomize_model(model=model, rng=123, return_state=True, pool=False)
    lid = sorted(trap_state["permuted_ids"].keys())[0]
    pids = np.asarray(trap_state["permuted_ids"][lid], dtype=int)
    W_perm = randomized_model.fc.weight.detach().cpu().numpy().copy()
    W_recon = unpermute_matrix(W_perm, pids)
    assert np.allclose(W_recon, original)
    W_perm2, pids2 = permute_matrix(original, rng=123)
    assert np.allclose(unpermute_matrix(W_perm2, pids2), original)
