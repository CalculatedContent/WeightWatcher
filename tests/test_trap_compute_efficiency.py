import inspect
import json
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import weightwatcher as ww
import weightwatcher.RMT_Util as rmt
import weightwatcher.weightwatcher as wwcore
import weightwatcher.remove_traps as rt


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


class SVDCallCounter:
    def __init__(self, real_fn):
        self.real_fn = real_fn
        self.calls = []

    def __call__(self, W, *args, **kwargs):
        arr = np.asarray(W)
        self.calls.append({"shape": tuple(arr.shape), "dtype": str(arr.dtype), "kwargs": dict(kwargs)})
        return self.real_fn(W, *args, **kwargs)

    def reset(self):
        self.calls.clear()

    @property
    def count(self):
        return len(self.calls)


def _normalize_selected(df):
    selected = df.copy()
    if "trap_index" in selected.columns and selected["trap_index"].min() == 0:
        selected["trap_index"] = selected["trap_index"].astype(int) + 1
    return selected


def _workflow(monkeypatch):
    model = OneLayer()
    watcher = ww.WeightWatcher(model=model)
    counter = SVDCallCounter(rmt.svd_full)
    monkeypatch.setattr(rmt, "svd_full", counter)
    monkeypatch.setattr(wwcore, "svd_full", counter)
    monkeypatch.setattr(rt, "svd_full", counter)
    return model, watcher, counter


def test_remove_traps_is_primary_public_ablation_api():
    assert hasattr(ww.WeightWatcher, "remove_traps")
    assert hasattr(ww.WeightWatcher, "randomize_model")
    assert hasattr(ww.WeightWatcher, "analyze_traps")
    assert not hasattr(ww.WeightWatcher, "move_traps")
    sig = inspect.signature(ww.WeightWatcher.remove_traps)
    for name in ["randomized_model", "trap_state", "trap_artifacts", "traps", "trap_indices"]:
        assert name in sig.parameters


def test_cached_remove_traps_does_not_call_svd_or_recollect(monkeypatch):
    _, watcher, counter = _workflow(monkeypatch)
    counter.reset()
    randomized_model, trap_state = watcher.randomize_model(model=watcher.model, rng=123, return_state=True, pool=False)
    assert counter.count == 0
    counter.reset()
    trap_df, trap_state = watcher.analyze_traps(randomized_model=randomized_model, layers=sorted(trap_state["permuted_ids"].keys()), trap_state=trap_state, permuted_ids=trap_state["permuted_ids"], return_artifacts=True, trap_burden=True, trap_burden_mode="fast", bulk_mode_sample=10, plot=False, pool=False)
    assert len(trap_df) > 0 and counter.count > 0
    monkeypatch.setattr(rt, "collect_trap_artifacts", lambda *a, **k: (_ for _ in ()).throw(AssertionError("cached remove_traps must not call collect_trap_artifacts")))
    selected = _normalize_selected(trap_df.iloc[[0]].copy())
    counter.reset()
    watcher.remove_traps(randomized_model=randomized_model, traps=selected, trap_state=trap_state, plot=False, pool=False)
    assert counter.count == 0


def test_repeated_cached_remove_traps_stays_zero_svd(monkeypatch):
    _, watcher, counter = _workflow(monkeypatch)
    randomized_model, trap_state = watcher.randomize_model(model=watcher.model, rng=123, return_state=True, pool=False)
    trap_df, trap_state = watcher.analyze_traps(randomized_model=randomized_model, trap_state=trap_state, return_artifacts=True, trap_burden=True, trap_burden_mode="fast", bulk_mode_sample=10, plot=False, pool=False)
    for i in range(min(3, len(trap_df))):
        selected = _normalize_selected(trap_df.iloc[[i]].copy())
        counter.reset()
        watcher.remove_traps(randomized_model=randomized_model, traps=selected, trap_state=trap_state, plot=False, pool=False)
        assert counter.count == 0


def test_compute_trace_records_svd_calls(tmp_path):
    model = OneLayer(); watcher = ww.WeightWatcher(model=model)
    with ww.ComputeTrace(enabled=True, log_path=str(tmp_path / "trace.jsonl")) as trace:
        randomized_model, trap_state = watcher.randomize_model(model=model, rng=123, return_state=True, pool=False)
        watcher.analyze_traps(randomized_model=randomized_model, trap_state=trap_state, return_artifacts=True, trap_burden=True, trap_burden_mode="fast", bulk_mode_sample=10, plot=False, pool=False)
    summary = trace.summary()
    assert summary["total_svd_calls"] >= 1
    svd_events = [e for e in trace.events if e.get("event_type") == "svd_full_end"]
    assert svd_events and all("matrix_shape" in e and "elapsed_ms" in e for e in svd_events)


def test_trace_jsonl_does_not_log_arrays(tmp_path):
    model = OneLayer(); watcher = ww.WeightWatcher(model=model)
    out = tmp_path / "trace.jsonl"
    with ww.ComputeTrace(enabled=True, log_path=str(out)) as trace:
        randomized_model, trap_state = watcher.randomize_model(model=model, rng=123, return_state=True, pool=False)
        trap_df, trap_state = watcher.analyze_traps(randomized_model=randomized_model, trap_state=trap_state, return_artifacts=True, trap_burden=True, trap_burden_mode="fast", bulk_mode_sample=10, plot=False, pool=False)
        watcher.remove_traps(randomized_model=randomized_model, traps=_normalize_selected(trap_df.iloc[[0]].copy()), trap_state=trap_state, plot=False, pool=False)
    for line in out.read_text().splitlines():
        evt = json.loads(line)
        for bad in ["W", "U", "Vh", "matrix", "weights", "T_perm"]:
            assert bad not in evt


def test_cached_remove_trace_has_zero_svd():
    model = OneLayer(); watcher = ww.WeightWatcher(model=model)
    randomized_model, trap_state = watcher.randomize_model(model=model, rng=123, return_state=True, pool=False)
    trap_df, trap_state = watcher.analyze_traps(randomized_model=randomized_model, trap_state=trap_state, return_artifacts=True, trap_burden=True, trap_burden_mode="fast", bulk_mode_sample=10, plot=False, pool=False)
    with ww.ComputeTrace(enabled=True) as trace:
        watcher.remove_traps(randomized_model=randomized_model, traps=_normalize_selected(trap_df.iloc[[0]].copy()), trap_state=trap_state, plot=False, pool=False)
    summary = trace.summary()
    assert summary["total_svd_calls"] == 0
    assert summary["collect_trap_artifacts_calls"] == 0


def test_fast_trace_has_no_full_bulk_or_original_basis():
    model = OneLayer(); watcher = ww.WeightWatcher(model=model)
    randomized_model, trap_state = watcher.randomize_model(model=model, rng=123, return_state=True, pool=False)
    with ww.ComputeTrace(enabled=True) as trace:
        watcher.analyze_traps(randomized_model=randomized_model, trap_state=trap_state, return_artifacts=True, trap_burden=True, trap_burden_mode="fast", bulk_mode_sample=10, plot=False, pool=False)
    summary = trace.summary()
    assert summary["full_bulk_metric_calls"] == 0
    assert summary["original_basis_metric_calls"] == 0
    assert summary["max_sampled_bulk_modes"] <= 10
