import pandas as pd
import numpy as np

from . import remove_traps as remove_traps_ops
from . import weightwatcher as wwcore
from .trap_histograms import plot_layer_trap_weight_histogram

def _lookup_layer_permuted_ids(layer_id, trap_state=None, permuted_ids=None):
    lid = int(layer_id)
    if isinstance(permuted_ids, dict):
        if lid in permuted_ids:
            return np.asarray(permuted_ids[lid], dtype=int)
        if str(lid) in permuted_ids:
            return np.asarray(permuted_ids[str(lid)], dtype=int)
    if isinstance(trap_state, dict):
        pid_map = trap_state.get("permuted_ids", {})
        if isinstance(pid_map, dict):
            if lid in pid_map:
                return np.asarray(pid_map[lid], dtype=int)
            if str(lid) in pid_map:
                return np.asarray(pid_map[str(lid)], dtype=int)
        layers = trap_state.get("layers", {})
        if isinstance(layers, dict):
            layer_state = layers.get(lid, layers.get(str(lid), None))
            if isinstance(layer_state, dict) and layer_state.get("permuted_ids") is not None:
                return np.asarray(layer_state["permuted_ids"], dtype=int)
    return None




def _sample_bulk_modes(svd_indices, eigenvalues, max_modes=None, seed=None, strategy="all"):
    ids = [int(i) for i in svd_indices]
    if max_modes is None or max_modes >= len(ids) or strategy == "all":
        return ids
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState(0)
    if strategy == "uniform":
        pick = rng.choice(ids, size=max_modes, replace=False)
        return sorted(int(x) for x in pick)
    if strategy == "stratified":
        ev = np.asarray([eigenvalues[i] for i in ids], dtype=float)
        q1, q2 = np.quantile(ev, [1/3, 2/3])
        bins = [[], [], []]
        for i,e in zip(ids, ev):
            bins[0 if e<=q1 else 1 if e<=q2 else 2].append(i)
        out=[]
        per=max(1, max_modes//3)
        for b in bins:
            if not b: continue
            k=min(len(b), per)
            out.extend(rng.choice(b,size=k,replace=False).tolist())
        remain=max_modes-len(out)
        if remain>0:
            rem=[i for i in ids if i not in out]
            if rem:
                out.extend(rng.choice(rem,size=min(remain,len(rem)),replace=False).tolist())
        return sorted(int(x) for x in out)
    raise ValueError("bulk_sampling_strategy must be one of: all, uniform, stratified")


def _build_trap_bulk_rows(layer_state, layer_rows, return_bulk_ids=False, bulk_only=False, trap_only=False, max_bulk_modes_per_layer=None, bulk_sampling_seed=None, bulk_sampling_strategy='all'):
    trap_svd = [int(i) for i in layer_state.get('trap_mode_indices_0based', [])]
    S = np.asarray(layer_state.get('S_perm', []), dtype=float)
    evals = S*S
    mp_max=float(layer_state.get('bulk_stats',{}).get('mp_bulk_max', np.nan))
    mp_min=float(layer_state.get('bulk_stats',{}).get('mp_bulk_min', 0.0))
    trap_set=set(trap_svd)
    inside=[i for i,e in enumerate(evals) if np.isfinite(mp_max) and e>=mp_min and e<=mp_max]
    bulk=[i for i in inside if i not in trap_set]
    bulk=_sample_bulk_modes(bulk, evals, max_bulk_modes_per_layer, bulk_sampling_seed, bulk_sampling_strategy)
    layer_state['trap_svd_indices']=trap_svd
    layer_state['bulk_svd_indices']=bulk
    layer_state['trap_id_to_svd_index']={i+1:v for i,v in enumerate(trap_svd)}
    layer_state['bulk_id_to_svd_index']={i+1:v for i,v in enumerate(bulk)}
    for r in layer_rows:
        r['mode_type']='trap'; r['ablation_type']='trap'; r['mode_id']=int(r.get('trap_index'))
        r['trap_id']=int(r.get('trap_index')); r['bulk_id']=np.nan; r['bulk_index']=np.nan
        r['is_trap']=True; r['is_bulk']=False
        r['svd_mode_index']=int(r.get('trap_mode_index_0based', r.get('trap_mode_index',-1)))
        ev=float(r.get('eval_perm', np.nan))
        r['eigenvalue']=ev; r['singular_value']=float(np.sqrt(ev)) if np.isfinite(ev) else np.nan
        r['mode_index']=r['svd_mode_index']; r['mp_lambda_min']=mp_min; r['mp_lambda_max']=mp_max
    bulk_rows=[]
    if return_bulk_ids:
        for bi,svd_i in enumerate(bulk, start=1):
            ev=float(evals[svd_i])
            bulk_rows.append({'layer_id':int(layer_state['layer_id']),'name':layer_state.get('name'),'longname':layer_state.get('longname'),'mode_type':'bulk','ablation_type':'bulk','mode_id':bi,'trap_id':np.nan,'trap_index':np.nan,'bulk_id':bi,'bulk_index':bi,'is_trap':False,'is_bulk':True,'svd_mode_index':svd_i,'mode_index':svd_i,'singular_value':float(S[svd_i]),'eigenvalue':ev,'eval_perm':ev,'mp_lambda_min':mp_min,'mp_lambda_max':mp_max,'is_inside_mp_bulk':True,'is_above_mp_edge':False,'is_below_mp_edge':False})
    if bulk_only: return bulk_rows
    if trap_only: return layer_rows
    return layer_rows + bulk_rows

def analyze_traps(
    watcher,
    model=None,
    layers=None,
    min_evals=wwcore.DEFAULT_MIN_EVALS,
    max_evals=wwcore.DEFAULT_MAX_EVALS,
    min_size=None,
    max_size=None,
    max_N=wwcore.DEFAULT_MAX_N,
    glorot_fix=False,
    plot=False,
    savefig=wwcore.DEF_SAVE_DIR,
    conv2d_norm=True,
    ww2x=wwcore.DEFAULT_WW2X,
    pool=wwcore.DEFAULT_POOL,
    conv2d_fft=False,
    fft=False,
    channels=None,
    svd_method=wwcore.FAST_SVD,
    start_ids=wwcore.DEFAULT_START_ID,
    base_model=None,
    peft=wwcore.DEFAULT_PEFT,
    rng=None,
    trap_burden=False,
    trap_burden_variant="top5",
    top_sector_l=1,
    trap_burden_mode="fast",
    compute_original_basis=None,
    compute_full_bulk_reference=None,
    bulk_mode_sample=10,
    compute_original_trap_svd=None,
    trap_state=None,
    return_artifacts=True,
    permuted_ids=None,
    already_randomized=False,
    return_bulk_ids=False,
    bulk_only=False,
    trap_only=False,
    max_bulk_modes_per_layer=None,
    bulk_sampling_seed=None,
    bulk_sampling_strategy="all",
):
    """Externalized implementation for WeightWatcher.analyze_traps()."""
    if layers is None:
        layers = []

    watcher.set_model_(model, base_model)

    if min_size or max_size:
        wwcore.logger.warning("min_size and max_size options changed to min_evals, max_evals, ignored for now")

    if ww2x:
        wwcore.logger.warning("WW2X option deprecated, reverting too POOL=False")
        ww2x = False
        pool = False

    params = wwcore.DEFAULT_PARAMS.copy()
    params[wwcore.MIN_EVALS] = min_evals
    params[wwcore.MAX_EVALS] = max_evals
    params[wwcore.MAX_N] = max_N

    params[wwcore.PLOT] = plot
    params[wwcore.RANDOMIZE] = True
    params[wwcore.MP_FIT] = True
    params[wwcore.GLOROT_FIT] = glorot_fix
    params[wwcore.CONV2D_NORM] = conv2d_norm

    params[wwcore.POOL] = pool
    params[wwcore.WW2X] = ww2x
    params[wwcore.CONV2D_FFT] = conv2d_fft
    params[wwcore.FFT] = fft

    params[wwcore.CHANNELS_STR] = channels
    params[wwcore.LAYERS] = layers
    params[wwcore.STACKED] = False

    params[wwcore.DETX] = False
    params[wwcore.SVD_METHOD] = svd_method
    params[wwcore.TOLERANCE] = wwcore.WEAK_RANK_LOSS_TOLERANCE
    params[wwcore.START_IDS] = start_ids

    params[wwcore.SAVEFIG] = savefig
    params[wwcore.PEFT] = peft
    params[wwcore.INVERSE] = False
    params["rng"] = remove_traps_ops._normalize_trap_rng(rng=rng)
    params["trap_burden"] = bool(trap_burden)
    params["trap_burden_variant"] = trap_burden_variant
    params["top_sector_l"] = int(top_sector_l)
    params["trap_burden_mode"] = trap_burden_mode
    params["bulk_mode_sample"] = 10 if bulk_mode_sample is None and trap_burden_mode == "fast" else bulk_mode_sample

    if compute_original_basis is None:
        compute_original_basis = (trap_burden_mode == "full")
    if compute_full_bulk_reference is None:
        compute_full_bulk_reference = (trap_burden_mode == "full")
    if compute_original_trap_svd is None:
        compute_original_trap_svd = (trap_burden_mode == "full")

    params["compute_original_basis"] = bool(compute_original_basis)
    params["compute_full_bulk_reference"] = bool(compute_full_bulk_reference)
    params["compute_original_trap_svd"] = bool(compute_original_trap_svd)

    wwcore.logger.debug("params {}".format(params))
    if not watcher.valid_params(params):
        msg = "Error, params not valid: \n {}".format(params)
        wwcore.logger.error(msg)
        raise Exception(msg)
    params = watcher.normalize_params(params)

    layer_iterator = watcher.make_layer_iterator(model=watcher.model, layers=layers, params=params, base_model=watcher.base_model)
    trap_rows = []
    trap_component_rows = []

    for ww_layer in layer_iterator:
        if not ww_layer.skipped and ww_layer.has_weights:
            watcher.apply_normalize_Wmats(ww_layer, params)

            if params[wwcore.FFT]:
                watcher.apply_FFT(ww_layer, params)

            layer_params = dict(params)
            layer_params["_keep_trap_matrix"] = bool(params.get(wwcore.PLOT, False))
            layer_params["return_artifacts"] = bool(return_artifacts)
            layer_params["already_randomized"] = bool(already_randomized)
            layer_perm_ids = _lookup_layer_permuted_ids(
                ww_layer.layer_id, trap_state=trap_state, permuted_ids=permuted_ids
            )
            if already_randomized:
                if layer_perm_ids is None:
                    raise ValueError(
                        f"Missing permute_ids for already-randomized layer_id={ww_layer.layer_id}. "
                        "Use layers=sorted(trap_state['permuted_ids'].keys()) and pass trap_state from randomize_model. "
                        "Also ensure randomize_model/analyze_traps use matching pool/start_ids/layers."
                    )
                layer_params["permuted_ids"] = layer_perm_ids
            elif layer_perm_ids is not None:
                layer_params["permuted_ids"] = layer_perm_ids
            layer_out = watcher.apply_analyze_traps(ww_layer, params=layer_params)
            if return_artifacts:
                layer_rows, layer_state = layer_out
                if trap_state is None:
                    trap_state = {"already_randomized": bool(already_randomized), "permuted_ids": {}, "layers": {}}
                trap_state.setdefault("layers", {})[int(ww_layer.layer_id)] = layer_state
                trap_state.setdefault("permuted_ids", {})[int(ww_layer.layer_id)] = layer_state.get("permuted_ids")
            else:
                layer_rows = layer_out
            if layer_rows or return_bulk_ids:
                layer_rows = _build_trap_bulk_rows(layer_state if return_artifacts else {"layer_id": int(ww_layer.layer_id), "name": ww_layer.name, "longname": ww_layer.longname, "S_perm": np.array([]), "trap_mode_indices_0based": []}, layer_rows or [], return_bulk_ids=return_bulk_ids, bulk_only=bulk_only, trap_only=trap_only, max_bulk_modes_per_layer=max_bulk_modes_per_layer, bulk_sampling_seed=bulk_sampling_seed, bulk_sampling_strategy=bulk_sampling_strategy)
                if params.get(wwcore.PLOT, False):
                    trap_infos = []
                    for row in layer_rows:
                        trap_idx_one_based = int(row.get("trap_index", -1))
                        trap_matrix = row.get("T_orig", None)
                        if trap_idx_one_based < 1 or trap_matrix is None:
                            continue

                        trap_infos.append(
                            {
                                "trap_index": trap_idx_one_based,
                                "trap_matrix": trap_matrix,
                                "trap_assessment": row.get("trap_assessment", "mixed"),
                                "trap_risk_score": row.get("trap_risk_score", 0.0),
                            }
                        )
                        trap_component_rows.append(
                            _top_trap_component_row(
                                row=row,
                                weight_matrix=ww_layer.Wmats[0],
                                top_k=10,
                            )
                        )

                    plot_layer_trap_weight_histogram(
                        ww_layer,
                        trap_infos,
                        params=params,
                        method_tag="analyze_traps",
                    )

                for row in layer_rows:
                    row.pop("T_orig", None)
                trap_rows.extend(layer_rows)

    if len(trap_rows) > 0:
        details = pd.DataFrame.from_records(trap_rows)
    else:
        details = pd.DataFrame(columns=watcher._trap_result_columns())

    trap_cols = watcher._trap_result_columns()
    details = details.reindex(columns=trap_cols + [c for c in details.columns if c not in trap_cols])

    if len(details) > 0:
        if "perm_mode_index" in details.columns:
            details["perm_mode_index_0based"] = details["perm_mode_index"].astype(int)
            details["perm_mode_index"] = details["perm_mode_index_0based"].apply(remove_traps_ops._internal_trap_index_to_api)
        if "trap_mode_index" in details.columns:
            details["trap_mode_index_0based"] = details["trap_mode_index"].astype(int)
            details["trap_mode_index"] = details["trap_mode_index_0based"].apply(remove_traps_ops._internal_trap_index_to_api)
        lead_cols = ["layer_id", "name"]
        details = details[lead_cols + [c for c in details.columns if c not in lead_cols]]

    watcher.details = details

    if len(trap_component_rows) > 0:
        trap_component_df = pd.DataFrame.from_records(trap_component_rows)
        fixed_cols = ["layer_id", "name", "trap_index", "trap_assessment", "trap_risk_score"]
        pair_cols = [col for i in range(1, 11) for col in (f"Wij_{i}", f"Cij_{i}")]
        ordered_cols = [c for c in (fixed_cols + pair_cols) if c in trap_component_df.columns]
        trap_component_df = trap_component_df.reindex(columns=ordered_cols)
        watcher.trap_component_summary = trap_component_df
        print(trap_component_df.to_string(index=False))
    else:
        watcher.trap_component_summary = pd.DataFrame()

    if return_artifacts and isinstance(trap_state, dict):
        for _lid, layer_state in trap_state.get("layers", {}).items():
            if not isinstance(layer_state, dict):
                continue
            modes0 = layer_state.get("trap_mode_indices_0based")
            if modes0 is None:
                modes_raw = layer_state.get("trap_mode_indices", [])
                if modes_raw:
                    modes_int = [int(x) for x in modes_raw]
                    if min(modes_int) >= 1:
                        modes0 = [m - 1 for m in modes_int]
                    else:
                        modes0 = modes_int
                else:
                    modes0 = []
            layer_state["trap_mode_indices_0based"] = [int(x) for x in modes0]
            layer_state["trap_mode_indices"] = remove_traps_ops._internal_trap_indices_to_api(layer_state["trap_mode_indices_0based"])

            for artifact in layer_state.get("artifacts", []) or []:
                if "trap_mode_index" in artifact:
                    artifact["trap_mode_index_0based"] = int(artifact["trap_mode_index"])
                    artifact["trap_mode_index"] = remove_traps_ops._internal_trap_index_to_api(artifact["trap_mode_index_0based"])
    return (details, trap_state) if return_artifacts else details


def _top_trap_component_row(row, weight_matrix, top_k=10):
    trap_matrix = np.asarray(row.get("T_orig", np.array([])), dtype=float)
    weight_matrix = np.asarray(weight_matrix, dtype=float)

    out = {
        "layer_id": row.get("layer_id"),
        "name": row.get("name"),
        "trap_index": int(row.get("trap_index", -1)),
        "trap_assessment": row.get("trap_assessment", "mixed"),
        "trap_risk_score": float(row.get("trap_risk_score", 0.0)),
    }

    for i in range(1, top_k + 1):
        out[f"Wij_{i}"] = np.nan
        out[f"Cij_{i}"] = np.nan

    if trap_matrix.size == 0 or trap_matrix.shape != weight_matrix.shape:
        return out

    flat_trap = trap_matrix.ravel()
    flat_weight = weight_matrix.ravel()
    abs_coeff = np.abs(flat_trap)
    max_abs = float(np.max(abs_coeff)) if abs_coeff.size else 0.0
    if max_abs <= 0.0:
        return out

    order = np.argsort(abs_coeff)[::-1][:top_k]
    for rank, idx in enumerate(order, start=1):
        out[f"Wij_{rank}"] = float(flat_weight[idx])
        out[f"Cij_{rank}"] = float(flat_trap[idx] / max_abs)

    return out
