import numpy as np
import pandas as pd

from weightwatcher.weightwatcher import WeightWatcher
from tests.test_remove_traps import _single_trap_setup, make_ww_layer


def test_randomized_model_workflow_public_api(monkeypatch):
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
    try:
        watcher.remove_traps(model={"dummy_weight": np.array([1.0])}, trap_state={"layers": {}}, traps=pd.DataFrame([{"trap_index": 1}]), plot=False)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "requires randomized_model" in str(e)
