import inspect
import pytest
import torch
import weightwatcher as ww

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
        watcher.remove_traps(model=model, traps=trap_df.iloc[[0]], trap_state=trap_state, plot=False)

def test_public_api_signatures():
    a = inspect.signature(ww.WeightWatcher.analyze_traps)
    for name in ["randomized_model","trap_state","return_artifacts","trap_burden_mode","bulk_mode_sample","compute_original_basis","compute_full_bulk_reference","compute_original_trap_svd"]:
        assert name in a.parameters
    assert "already_randomized" not in a.parameters
    r = inspect.signature(ww.WeightWatcher.remove_traps)
    for name in ["randomized_model","trap_state","trap_artifacts"]:
        assert name in r.parameters
    assert "already_randomized" not in r.parameters
