import copy
import hashlib
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import remove_traps as remove_traps_ops
from .RMT_Util import unpermute_matrix
from .constants import DEFAULT_PARAMS, DEFAULT_START_ID, FAST_SVD, LAYERS, PEFT, PLOT, POOL, SVD_METHOD, WW_NAME
from .weightwatcher import WeightWatcher

logger = logging.getLogger(WW_NAME)


@dataclass
class TrapAnalysisBundle:
    checkpoint_id: Optional[str]
    layer_id: int
    layer_name: str
    layer_longname: str
    W_orig: np.ndarray
    W_perm: np.ndarray
    permute_ids: np.ndarray
    permute_mode: str
    seed: Optional[int]
    rng_state: Optional[dict]
    permute_fingerprint: str
    U_perm: np.ndarray
    S_perm: np.ndarray
    Vh_perm: np.ndarray
    trap_metrics: pd.DataFrame
    trap_mode_map: Dict[int, int]
    mp_bulk_edge: Dict[str, float] = field(default_factory=dict)
    bundle_path: Optional[str] = None


def _fingerprint_perm(permute_ids: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(permute_ids).tobytes()).hexdigest()


def _bundle_filename(bundle: TrapAnalysisBundle) -> str:
    ck = str(bundle.checkpoint_id or "na")
    seed = "none" if bundle.seed is None else str(bundle.seed)
    return f"trap_bundle_step_{ck}_layer_{bundle.layer_id}_{bundle.permute_fingerprint}_seed_{seed}.pkl"


def save_trap_bundle(bundle: TrapAnalysisBundle, bundle_dir: str) -> str:
    os.makedirs(bundle_dir, exist_ok=True)
    path = os.path.join(bundle_dir, _bundle_filename(bundle))
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    bundle.bundle_path = path
    return path


def load_trap_bundle(path: str) -> TrapAnalysisBundle:
    with open(path, "rb") as f:
        return pickle.load(f)


def analyze_traps_bundle(model_or_watcher, layers=None, save_bundle=False, bundle_dir=None, return_bundle=True, checkpoint_id=None, seed=None, rng=None, plot=False, pool=True):
    watcher = model_or_watcher if isinstance(model_or_watcher, WeightWatcher) else WeightWatcher(model=model_or_watcher)
    model = watcher.model if isinstance(model_or_watcher, WeightWatcher) else model_or_watcher
    params = DEFAULT_PARAMS.copy()
    params[POOL] = pool
    params[PLOT] = plot
    params[LAYERS] = [] if layers is None else layers
    params[SVD_METHOD] = FAST_SVD
    params[PEFT] = params.get(PEFT)
    params["seed"] = seed
    params["rng"] = remove_traps_ops._normalize_trap_rng(rng=rng, seed=seed)
    params = watcher.normalize_params(params)

    trap_df = watcher.analyze_traps(model=model, layers=layers or [], plot=plot, pool=pool, rng=rng if rng is not None else seed)
    bundles = {}
    rows = []
    layer_iterator = watcher.make_layer_iterator(model=watcher.model, layers=layers or [], params=params, base_model=watcher.base_model)
    for ww_layer in layer_iterator:
        if ww_layer.skipped or not ww_layer.has_weights:
            continue
        layer_traps = trap_df[trap_df["layer_id"].astype(int) == int(ww_layer.layer_id)].copy()
        if len(layer_traps) == 0:
            continue
        analysis_layer = ww_layer.copy()
        analysis_layer.Wmats = [ww_layer.Wmats[0].copy()]
        analysis_layer.w_norm = 1.0
        analysis_layer.permute_ids = []
        watcher.apply_normalize_Wmats(analysis_layer, params)
        watcher.apply_permute_W(analysis_layer, params, rng=params["rng"])
        W_perm = analysis_layer.Wmats[0].copy()
        permute_ids = np.asarray(analysis_layer.permute_ids[0]).copy()
        fp = _fingerprint_perm(permute_ids)
        U, S, Vh = remove_traps_ops.svd_full(W_perm)

        layer_traps["permute_fingerprint"] = fp
        layer_traps["bundle_id"] = f"{checkpoint_id}:{ww_layer.layer_id}:{fp}"
        trap_mode_map = {int(r.trap_index): int(r.trap_mode_index) for r in layer_traps.itertuples()}
        bundle = TrapAnalysisBundle(
            checkpoint_id=str(checkpoint_id) if checkpoint_id is not None else None,
            layer_id=int(ww_layer.layer_id),
            layer_name=str(getattr(ww_layer, "name", "")),
            layer_longname=str(getattr(ww_layer, "longname", "")),
            W_orig=np.asarray(ww_layer.Wmats[0]).copy(),
            W_perm=W_perm,
            permute_ids=permute_ids,
            permute_mode="shuffle",
            seed=seed,
            rng_state=params["rng"].get_state() if params.get("rng") is not None else None,
            permute_fingerprint=fp,
            U_perm=U,
            S_perm=S,
            Vh_perm=Vh,
            trap_metrics=layer_traps.copy(),
            trap_mode_map=trap_mode_map,
            mp_bulk_edge={"mp_bulk_max": float(layer_traps["mp_bulk_max"].iloc[0]) if "mp_bulk_max" in layer_traps.columns else np.nan},
        )
        if save_bundle:
            path = save_trap_bundle(bundle, bundle_dir or "trap_bundles")
            layer_traps["bundle_path"] = path
            bundle.bundle_path = path
        bundles[int(ww_layer.layer_id)] = bundle
        rows.append(layer_traps)
        logger.info(f"trap bundle layer_id={ww_layer.layer_id} traps={len(layer_traps)} permute_fingerprint={fp} bundle_path={bundle.bundle_path}")

    out_df = pd.concat(rows, ignore_index=True) if rows else trap_df.copy()
    return (out_df, bundles) if return_bundle else (out_df, None)


def remove_single_trap_from_bundle(model, bundle: TrapAnalysisBundle, trap_row, inplace=False, allow_model_mismatch=False, atol=1e-8, rtol=1e-5):
    row = trap_row if isinstance(trap_row, dict) else trap_row.to_dict()
    if int(row["layer_id"]) != int(bundle.layer_id):
        raise ValueError("layer_id mismatch between trap row and bundle")
    if str(row.get("permute_fingerprint")) != str(bundle.permute_fingerprint):
        raise ValueError("permute_fingerprint mismatch between trap row and bundle")
    tmi = int(row["trap_mode_index"])
    if tmi < 0 or tmi >= len(bundle.S_perm):
        raise ValueError("trap_mode_index not found in bundle decomposition")
    if "sigma_perm" in row and not np.isclose(float(row["sigma_perm"]), float(bundle.S_perm[tmi]), rtol=rtol, atol=atol):
        raise ValueError("sigma_perm mismatch between trap row and bundle decomposition")

    watcher = WeightWatcher(model=model)
    params = watcher.normalize_params(DEFAULT_PARAMS.copy())
    target_layer = None
    for ww_layer in watcher.make_layer_iterator(model=watcher.model, layers=[bundle.layer_id], params=params, base_model=watcher.base_model):
        if int(ww_layer.layer_id) == int(bundle.layer_id):
            target_layer = ww_layer
            break
    if target_layer is None:
        raise ValueError("target layer not found in model")
    current_W = np.asarray(target_layer.Wmats[0])
    if current_W.shape != bundle.W_orig.shape:
        raise ValueError("current model layer shape mismatch vs bundle")
    if (not allow_model_mismatch) and (not np.allclose(current_W, bundle.W_orig, rtol=rtol, atol=atol)):
        raise ValueError("current model layer weights do not match bundle original weights")

    u = bundle.U_perm[:, tmi]
    v = bundle.Vh_perm[tmi, :]
    T_perm = bundle.S_perm[tmi] * np.outer(u, v)
    W_perm_abl = bundle.W_perm - T_perm
    W_abl = unpermute_matrix(W_perm_abl, bundle.permute_ids)

    out_model = model if inplace else copy.deepcopy(model)
    out_watcher = WeightWatcher(model=out_model)
    out_params = out_watcher.normalize_params(DEFAULT_PARAMS.copy())
    for l in out_watcher.make_layer_iterator(model=out_watcher.model, layers=[bundle.layer_id], params=out_params, base_model=out_watcher.base_model):
        if int(l.layer_id) == int(bundle.layer_id):
            out_watcher.replace_layer_weights(l.layer_id, l.framework_layer, W_abl)
            break
    meta = {"ok": True, "layer_id": bundle.layer_id, "trap_index": int(row.get("trap_index", -1)), "trap_mode_index": tmi, "permute_fingerprint": bundle.permute_fingerprint}
    logger.info(f"bundle ablation checkpoint={bundle.checkpoint_id} layer_id={bundle.layer_id} trap_index={meta['trap_index']} trap_mode_index={tmi} permute_fingerprint={bundle.permute_fingerprint} verification passed")
    return out_model, meta


def remove_traps_from_bundle(model, bundle: TrapAnalysisBundle, trap_indices=None, trap_mode_indices=None, inplace=False, allow_model_mismatch=False):
    rows = bundle.trap_metrics.copy()
    if trap_indices is not None:
        rows = rows[rows["trap_index"].isin(list(trap_indices))]
    if trap_mode_indices is not None:
        rows = rows[rows["trap_mode_index"].isin(list(trap_mode_indices))]
    if len(rows) == 0:
        raise ValueError("No traps selected from bundle")
    out_model = model
    metas = []
    for _, row in rows.iterrows():
        out_model, meta = remove_single_trap_from_bundle(out_model, bundle, row, inplace=True, allow_model_mismatch=allow_model_mismatch)
        metas.append(meta)
    return (out_model if inplace else copy.deepcopy(out_model)), pd.DataFrame.from_records(metas)


def run_trap_bundle_ablation_experiment(model, checkpoint_step, checkpoint_path, layers, evaluate_fn, bulk_baseline_fn=None, bundle_dir=None, save_bundles=True):
    trap_df, bundles = analyze_traps_bundle(model, layers=layers, save_bundle=save_bundles, bundle_dir=bundle_dir, checkpoint_id=checkpoint_step, return_bundle=True, plot=False)
    base = evaluate_fn(model)
    out_rows = []
    for _, row in trap_df.iterrows():
        layer_id = int(row["layer_id"])
        bundle = bundles[layer_id]
        rec = dict(row)
        rec.update({"checkpoint_step": checkpoint_step, "checkpoint_path": checkpoint_path, "ok": False, "error": None, "base_train_accuracy": base.get("train_accuracy"), "base_test_accuracy": base.get("test_accuracy")})
        try:
            ablated_model, meta = remove_single_trap_from_bundle(model, bundle, row, inplace=False)
            res = evaluate_fn(ablated_model)
            rec["trap_train_accuracy"] = res.get("train_accuracy")
            rec["trap_test_accuracy"] = res.get("test_accuracy")
            rec["trap_delta_train_accuracy"] = rec["trap_train_accuracy"] - rec["base_train_accuracy"]
            rec["trap_delta_test_accuracy"] = rec["trap_test_accuracy"] - rec["base_test_accuracy"]
            if bulk_baseline_fn is not None:
                bulk = bulk_baseline_fn(model, bundle, row)
                rec.update(bulk)
                if "bulk_delta_test_accuracy_mean" in rec:
                    rec["trap_damage_excess_vs_bulk"] = rec["trap_delta_test_accuracy"] - rec["bulk_delta_test_accuracy_mean"]
            rec["ok"] = bool(meta.get("ok", False))
            rec["bundle_path"] = bundle.bundle_path
        except Exception as exc:
            rec["error"] = str(exc)
        out_rows.append(rec)
    return pd.DataFrame.from_records(out_rows)
