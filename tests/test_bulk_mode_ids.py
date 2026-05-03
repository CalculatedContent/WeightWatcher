import numpy as np
import pytest

torch = pytest.importorskip('torch')
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

def test_bulk_ids_returned_and_start_at_one():
    model = OneLayer()
    watcher = ww.WeightWatcher(model=model)
    randomized_model, state = watcher.randomize_model(model=model, rng=123, return_state=True)
    df, state = watcher.analyze_traps(randomized_model=randomized_model, trap_state=state, return_artifacts=True, return_bulk_ids=True, plot=False)
    assert 'mode_type' in df.columns
    assert ((df[df['mode_type']=='trap']['trap_id'].dropna() >= 1).all())
    bulk = df[df['mode_type']=='bulk']
    assert len(bulk) > 0
    assert (bulk['bulk_id'] >= 1).all()
    for lid, layer_state in state['layers'].items():
        assert set(layer_state.get('trap_svd_indices', [])).isdisjoint(set(layer_state.get('bulk_svd_indices', [])))

def test_bulk_sampling_deterministic_and_remove_bulk_changes_layer_only():
    model = OneLayer()
    watcher = ww.WeightWatcher(model=model)
    randomized_model, state = watcher.randomize_model(model=model, rng=123, return_state=True)
    df1, state1 = watcher.analyze_traps(randomized_model=randomized_model, trap_state=state, return_artifacts=True, return_bulk_ids=True, max_bulk_modes_per_layer=5, bulk_sampling_seed=123, plot=False)
    df2, _ = watcher.analyze_traps(randomized_model=randomized_model, trap_state=state, return_artifacts=True, return_bulk_ids=True, max_bulk_modes_per_layer=5, bulk_sampling_seed=123, plot=False)
    b1 = df1[df1['mode_type']=='bulk'][['layer_id','bulk_id']].sort_values(['layer_id','bulk_id']).reset_index(drop=True)
    b2 = df2[df2['mode_type']=='bulk'][['layer_id','bulk_id']].sort_values(['layer_id','bulk_id']).reset_index(drop=True)
    assert b1.equals(b2)
    lid = int(b1.iloc[0]['layer_id'])
    bid = int(b1.iloc[0]['bulk_id'])
    old = randomized_model.fc.weight.detach().clone()
    out = watcher.remove_modes(mode_ids_by_layer={lid:[bid]}, mode_type='bulk', randomized_model=randomized_model, trap_state=state1, plot=False)
    assert not torch.allclose(old, out.fc.weight.detach())


def test_invalid_bulk_id_errors():
    model = OneLayer()
    watcher = ww.WeightWatcher(model=model)
    randomized_model, state = watcher.randomize_model(model=model, rng=123, return_state=True)
    _, state = watcher.analyze_traps(randomized_model=randomized_model, trap_state=state, return_artifacts=True, return_bulk_ids=True, plot=False)
    with pytest.raises(ValueError):
        watcher.remove_modes(mode_ids_by_layer={999:[1]}, mode_type='bulk', randomized_model=randomized_model, trap_state=state, plot=False)
