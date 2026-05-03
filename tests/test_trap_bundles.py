import numpy as np
import pandas as pd
import pytest

import weightwatcher as ww
from weightwatcher.trap_bundles import analyze_traps_bundle, load_trap_bundle, remove_single_trap_from_bundle

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


pytestmark = pytest.mark.skipif(torch is None, reason="torch required")


if torch is not None:
    class TinyTrapNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 12, bias=False)
            with torch.no_grad():
                u = torch.linspace(1.0, 2.0, steps=12)
                v = torch.linspace(-2.0, 1.0, steps=16)
                self.fc1.weight.copy_(35.0 * torch.outer(u, v))


def test_analyze_traps_bundle_roundtrip(tmp_path):
    model = TinyTrapNet()
    watcher = ww.WeightWatcher(model=model)
    trap_df, bundles = analyze_traps_bundle(watcher, save_bundle=True, bundle_dir=str(tmp_path), checkpoint_id="1")
    assert isinstance(trap_df, pd.DataFrame)
    if len(trap_df) == 0 or len(bundles) == 0:
        pytest.skip("no traps")
    row = trap_df.iloc[0]
    b = bundles[int(row.layer_id)]
    assert row.permute_fingerprint == b.permute_fingerprint
    assert "bundle_path" in trap_df.columns
    loaded = load_trap_bundle(row.bundle_path)
    assert loaded.permute_fingerprint == b.permute_fingerprint


def test_remove_single_trap_from_bundle_independent_of_seed():
    model = TinyTrapNet()
    watcher = ww.WeightWatcher(model=model)
    trap_df, bundles = analyze_traps_bundle(watcher, seed=111)
    if len(trap_df) == 0:
        pytest.skip("no traps")
    row = trap_df.iloc[0]
    b = bundles[int(row.layer_id)]

    row = row.copy()
    row["sigma_perm"] = float(b.S_perm[int(row["trap_mode_index"])])
    m1, meta1 = remove_single_trap_from_bundle(model, b, row, allow_model_mismatch=True)
    # changing seed in fresh analysis should not affect prior bundle-based ablation
    _df2, _bundles2 = analyze_traps_bundle(ww.WeightWatcher(model=TinyTrapNet()), seed=999)
    m2, meta2 = remove_single_trap_from_bundle(model, b, row, allow_model_mismatch=True)
    assert meta1["permute_fingerprint"] == meta2["permute_fingerprint"]


def test_remove_single_trap_mismatch_raises():
    model = TinyTrapNet()
    trap_df, bundles = analyze_traps_bundle(ww.WeightWatcher(model=model), seed=123)
    if len(trap_df) == 0:
        pytest.skip("no traps")
    row = trap_df.iloc[0].copy()
    b = bundles[int(row.layer_id)]
    row["permute_fingerprint"] = "bad"
    with pytest.raises(ValueError, match="permute_fingerprint mismatch"):
        remove_single_trap_from_bundle(model, b, row)


def test_legacy_api_still_works():
    model = TinyTrapNet()
    watcher = ww.WeightWatcher(model=model)
    randomized_model, trap_state = watcher.randomize_model(model=model, rng=1337, return_state=True)
    df = watcher.analyze_traps(randomized_model=randomized_model, trap_state=trap_state, plot=False, savefig=False, return_artifacts=False)
    assert isinstance(df, pd.DataFrame)
