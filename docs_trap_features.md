# Correlation Trap Workflow (`analyze_traps` and `remove_traps`)

This guide explains how to use the new correlation-trap workflow that was recently merged into WeightWatcher.

> ⚠️ **Status note:** these features were **vibe coded** and are **not yet extensively tested**. Please validate outputs on your own models and use caution before applying them in production pipelines.

## What these features do

- `analyze_traps(...)` inspects selected layers and reports candidate correlation-trap modes.
- `remove_traps(...)` removes selected trap modes from those layers and returns an updated model.
- `randomize_model(...)` randomizes once and returns reusable state for faster ablation loops.

These are intended as **public WeightWatcher APIs** on `ww.WeightWatcher`.

Use this flow when you suspect a layer looks random except for isolated spikes (classic trap signature).

## 1) Analyze trap candidates

```python
import weightwatcher as ww
import torchvision.models as models

model = models.vgg19_bn(pretrained=True)
watcher = ww.WeightWatcher(model=model)

trap_df = watcher.analyze_traps(
    layers=[3, 5],
    plot=True,
    savefig="trap_images",
    rng=123,
)

print(trap_df[["layer_id", "layer_name", "num_traps"]])
```

### Tips

- Start with a small set of layers (`layers=[...]`) you already flagged with `analyze(randomize=True)`.
- Set a fixed `rng` (int seed) for reproducibility.
- Use `plot=True` + `savefig=...` to inspect before/after spectra artifacts.

## 2) Remove selected trap indices

After identifying trap indices of interest, run:

```python
clean_model = watcher.remove_traps(
    model=model,
    layers=[3, 5],
    trap_indices=[1],
    seed=123,
    pool=True,
    plot=False,
)
```

### Important behavior

- `trap_indices` are **1-based trap IDs** reported by the trap analysis flow.
- Current implementation is focused on supported layer/matrix paths used by the merged feature tests.
- Always compare metrics before and after (`analyze`, `get_summary`, eval task metrics).

## 3) Recommended validation checklist

Because this workflow is new and still stabilizing:

1. Save baseline model metrics and downstream task scores.
2. Run `analyze_traps(...)` with a fixed seed and inspect plots.
3. Remove one trap at a time first (`trap_indices=[1]`, then iterate).
4. Re-run WeightWatcher metrics and downstream evaluation.
5. Keep a rollback path (checkpointed original model).

## Minimal end-to-end sketch

```python
watcher = ww.WeightWatcher(model=model)

# Baseline
baseline_details = watcher.analyze(plot=False)
baseline_summary = watcher.get_summary(baseline_details)

# Trap diagnostics
trap_df = watcher.analyze_traps(layers=[3, 5], rng=123, plot=True, savefig="trap_images")

# Trap removal (example: first detected trap mode)
clean_model = watcher.remove_traps(model=model, layers=[3, 5], trap_indices=[1], seed=123)

# Re-check
clean_watcher = ww.WeightWatcher(model=clean_model)
clean_details = clean_watcher.analyze(plot=False)
clean_summary = clean_watcher.get_summary(clean_details)
```

If you find edge cases, please open an issue with model type, layer selection, and seed used.

## Targeted unit tests

Use this exact command for the trap-analysis/trap-removal tests:

```bash
pytest -q tests/test_analyze_traps.py tests/test_remove_traps.py
```

> Note: the second path is `test_remove_traps.py` (not `.pyz`).

## Fast randomized trap ablation workflow

Use the new cached workflow to randomize once, analyze once, then remove traps without re-randomizing:

```python
randomized_model, trap_state = watcher.randomize_model(
    model=model, layers=layers, rng=seed, return_state=True, pool=False
)
permuted_ids = trap_state["permuted_ids"]
randomized_layers = sorted(permuted_ids.keys())

trap_df, trap_state = watcher.analyze_traps(
    randomized_model=randomized_model,
    layers=randomized_layers,
    trap_state=trap_state,
    permuted_ids=permuted_ids,
    return_artifacts=True,
    trap_burden=True,
    trap_burden_mode="fast",
    bulk_mode_sample=10,
    plot=False,
    pool=False,
)
ablated_model = watcher.remove_traps(
    randomized_model=randomized_model,
    traps=trap_df.iloc[[0]],
    trap_state=trap_state,
    plot=False,
    pool=False,
)
```

Warning: `trap_burden_mode="fast"` uses approximate overlap and bulk-reference metrics for speed. Use `trap_burden_mode="full"` for expensive original-basis diagnostics.

### Why this is faster (and how to avoid long runs)

- Reuse `randomized_model` + `trap_state` across iterative removals instead of re-running full randomization.
- Set `return_artifacts=True` in `analyze_traps(...)` and pass that same `trap_state` into `remove_traps(...)` so trap artifacts are cached and reused.
- Use matching `pool/start_ids/layers` between `randomize_model(...)` and `analyze_traps(...)`.
- Use `layers=sorted(trap_state["permuted_ids"].keys())` and do **not** analyze non-randomized layers in cached mode.
- Cached workflow (`trap_state`/`permuted_ids`/`return_artifacts`) requires `randomized_model=...`.
- Use `trap_burden_mode="fast"` and a small `bulk_mode_sample` during exploration; switch to `"full"` only for final verification.
- Restrict analysis to a small `layers=[...]` subset first, then expand after confirming expected behavior.
