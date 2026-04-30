import pandas as pd
import numpy as np

from . import remove_traps as remove_traps_ops
from . import weightwatcher as wwcore
from .trap_histograms import plot_layer_trap_weight_histogram


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
    plot=True,
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
    permuted_ids=None,
    trap_state=None,
    already_randomized=False,
    return_artifacts=False,
    trap_burden=False,
    trap_burden_variant="top5",
    top_sector_l=1,
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
    params["already_randomized"] = bool(already_randomized)

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
            if trap_state is not None and isinstance(trap_state, dict) and "permuted_ids" in trap_state:
                layer_params["permuted_ids"] = trap_state.get("permuted_ids", {})
            elif permuted_ids is not None:
                layer_params["permuted_ids"] = permuted_ids
            layer_params["already_randomized"] = bool(already_randomized)
            layer_params["_keep_trap_matrix"] = bool(params.get(wwcore.PLOT, False))
            layer_rows = watcher.apply_analyze_traps(ww_layer, params=layer_params)
            if layer_rows:
                if params.get(wwcore.PLOT, False):
                    trap_infos = []
                    for row in layer_rows:
                        trap_idx_zero_based = int(row.get("trap_index", -1))
                        trap_matrix = row.get("T_orig", None)
                        if trap_idx_zero_based < 0 or trap_matrix is None:
                            continue

                        trap_infos.append(
                            {
                                "trap_index": trap_idx_zero_based + 1,
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

    if return_artifacts:
        out_state = trap_state.copy() if isinstance(trap_state, dict) else {}
        out_state.setdefault("permuted_ids", permuted_ids if permuted_ids is not None else {})
        out_state["details_rows"] = details.to_dict(orient="records")
        out_state["already_randomized"] = bool(already_randomized)
        return details, out_state
    return details


def _top_trap_component_row(row, weight_matrix, top_k=10):
    trap_matrix = np.asarray(row.get("T_orig", np.array([])), dtype=float)
    weight_matrix = np.asarray(weight_matrix, dtype=float)

    out = {
        "layer_id": row.get("layer_id"),
        "name": row.get("name"),
        "trap_index": int(row.get("trap_index", -1)) + 1,
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
