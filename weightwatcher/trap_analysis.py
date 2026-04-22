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
    params["top_sector_l"] = int(top_sector_l)
    if int(top_sector_l) < 1:
        raise ValueError("top_sector_l must be >= 1")

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
    if len(details) > 0 and "trap_variance_burden" in details.columns:
        details["layer_trap_variance_burden"] = (
            details.groupby("layer_id")["trap_variance_burden"].transform("sum")
        )

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

    return details


def compute_original_basis_for_traps(watcher, ww_layer, params=None):
    if params is None:
        params = wwcore.DEFAULT_PARAMS.copy()
    if len(ww_layer.Wmats) != 1:
        return None

    W_true = ww_layer.Wmats[0].astype(float)
    U0, S0, V0h = wwcore.svd_full(W_true, method=params[wwcore.SVD_METHOD])
    return {
        "W_true": W_true,
        "U0": U0,
        "S0": S0,
        "V0": V0h.T,
    }


def compute_trap_delta(eval_perm, mp_bulk_max):
    eval_perm = float(eval_perm)
    mp_bulk_max = float(mp_bulk_max)
    if (not np.isfinite(eval_perm)) or (not np.isfinite(mp_bulk_max)) or mp_bulk_max <= 0.0:
        return float(np.nan)
    return float(max(eval_perm - mp_bulk_max, 0.0) / mp_bulk_max)


def compute_trap_ipr_q(vec):
    v = np.asarray(vec, dtype=float).ravel()
    if v.size == 0:
        return float(np.nan), float(np.nan)
    norm = np.linalg.norm(v)
    if (not np.isfinite(norm)) or norm <= 0.0:
        return float(np.nan), float(np.nan)

    v = v / norm
    ipr = float(np.sum(v ** 4))
    m = int(len(v))
    if m <= 1:
        q = 1.0
    else:
        q = (m * ipr - 1.0) / (m - 1.0)
        q = float(np.clip(q, 0.0, 1.0))
    return ipr, float(q)


def compute_top_sector_overlap(overlaps, top_sector_l=1):
    ell = int(top_sector_l)
    if ell < 1:
        raise ValueError("top_sector_l must be >= 1")

    overlap_vec = np.asarray(overlaps, dtype=float).ravel()
    if overlap_vec.size == 0:
        return float(np.nan), 0

    ell_eff = min(ell, int(len(overlap_vec)))
    return float(np.sum(overlap_vec[:ell_eff])), int(ell_eff)


def compute_trap_variance_burden(trap_delta, trap_q, trap_top_sector_overlap):
    trap_delta = float(trap_delta)
    trap_q = float(trap_q)
    trap_top_sector_overlap = float(trap_top_sector_overlap)
    if (not np.isfinite(trap_delta)) or (not np.isfinite(trap_q)) or (not np.isfinite(trap_top_sector_overlap)):
        return float(np.nan)
    return float((trap_delta ** 2) * trap_q * (trap_top_sector_overlap ** 2))


def analyze_single_trap(watcher, ww_layer, trap_mode_index, original_basis_cache=None, params=None, trap_index=0):
    if params is None:
        params = wwcore.DEFAULT_PARAMS.copy()
    if original_basis_cache is None:
        original_basis_cache = compute_original_basis_for_traps(watcher, ww_layer, params=params)

    W_perm = ww_layer.Wmats[0].astype(float)
    p_ids = ww_layer.permute_ids[0]

    U_perm, S_perm, Vh_perm = wwcore.svd_full(W_perm, method=params[wwcore.SVD_METHOD])
    V_perm = Vh_perm.T

    sigma_perm = float(S_perm[trap_mode_index])
    u_perm = U_perm[:, trap_mode_index]
    v_perm = V_perm[:, trap_mode_index]

    T_perm = sigma_perm * np.outer(u_perm, v_perm)
    T_orig = wwcore.unpermute_matrix(T_perm, p_ids)

    Ut, St, Vht = wwcore.svd_full(T_orig, method=params[wwcore.SVD_METHOD])
    u_trap = Ut[:, 0]
    v_trap = Vht.T[:, 0]

    U0 = original_basis_cache["U0"]
    V0 = original_basis_cache["V0"]

    left_overlaps = np.abs(U0.T @ u_trap) ** 2
    right_overlaps = np.abs(V0.T @ v_trap) ** 2

    left_top_mode = int(np.argmax(left_overlaps))
    right_top_mode = int(np.argmax(right_overlaps))
    left_top_mass = float(np.max(left_overlaps))
    right_top_mass = float(np.max(right_overlaps))

    eps = 1e-12
    left_overlap_entropy = float(-np.sum((left_overlaps + eps) * np.log(left_overlaps + eps)))
    right_overlap_entropy = float(-np.sum((right_overlaps + eps) * np.log(right_overlaps + eps)))
    left_overlap_ipr = float(np.sum(left_overlaps ** 2))
    right_overlap_ipr = float(np.sum(right_overlaps ** 2))

    st_sq = St * St
    rank1_mass_after_unpermute = float(st_sq[0] / (np.sum(st_sq) + eps))

    u_metrics = watcher._trap_vector_metrics(u_trap)
    v_metrics = watcher._trap_vector_metrics(v_trap)
    u_oi = watcher._trap_vector_order_invariant_stats(u_trap)
    v_oi = watcher._trap_vector_order_invariant_stats(v_trap)

    eval_perm = sigma_perm ** 2
    top_sector_l = int(params.get("top_sector_l", 1))
    trap_delta = compute_trap_delta(eval_perm=eval_perm, mp_bulk_max=ww_layer.bulk_max)
    trap_ipr, trap_q = compute_trap_ipr_q(v_perm)
    trap_top_sector_overlap, top_sector_l_effective = compute_top_sector_overlap(
        right_overlaps,
        top_sector_l=top_sector_l,
    )
    trap_variance_burden = compute_trap_variance_burden(
        trap_delta=trap_delta,
        trap_q=trap_q,
        trap_top_sector_overlap=trap_top_sector_overlap,
    )
    trap_result = {
        "layer_id": ww_layer.layer_id,
        "name": ww_layer.name,
        "longname": ww_layer.longname,
        "layer_type": str(ww_layer.the_type),
        "N": ww_layer.N,
        "M": ww_layer.M,
        "rf": ww_layer.rf,
        "Q": ww_layer.N / ww_layer.M if ww_layer.M > 0 else np.nan,
        "trap_index": int(trap_index),
        "perm_mode_index": int(trap_mode_index),
        "sigma_perm": sigma_perm,
        "eval_perm": float(eval_perm),
        "mp_bulk_max": float(ww_layer.bulk_max),
        "mp_bulk_min": float(ww_layer.bulk_min),
        "sigma_mp": float(ww_layer.sigma_mp),
        "num_spikes": int(ww_layer.num_spikes),
        "rank1_mass_after_unpermute": rank1_mass_after_unpermute,
        "sigma_trap_top": float(St[0]),
        "left_top_mode": left_top_mode,
        "right_top_mode": right_top_mode,
        "left_top_mass": left_top_mass,
        "right_top_mass": right_top_mass,
        "left_overlap_entropy": left_overlap_entropy,
        "right_overlap_entropy": right_overlap_entropy,
        "left_overlap_ipr": left_overlap_ipr,
        "right_overlap_ipr": right_overlap_ipr,
        "trap_detected": True,
        "trap_eval_minus_bulk": float(eval_perm - ww_layer.bulk_max),
        # Paper-aligned trap metrics (NeurIPS trap paper definitions).
        "trap_delta": trap_delta,
        "trap_ipr": trap_ipr,
        "trap_q": trap_q,
        "trap_diffuseness": float(1.0 - trap_q) if np.isfinite(trap_q) else np.nan,
        "top_sector_l": top_sector_l,
        "top_sector_l_effective": top_sector_l_effective,
        "trap_top_sector_overlap": trap_top_sector_overlap,
        "trap_variance_burden": trap_variance_burden,
    }

    for k, v in u_metrics.items():
        trap_result[f"u_{k}"] = v
    for k, v in v_metrics.items():
        trap_result[f"v_{k}"] = v
    for k, v in u_oi.items():
        trap_result[f"u_{k}"] = v
    for k, v in v_oi.items():
        trap_result[f"v_{k}"] = v

    trap_result["trap_balance_ratio"] = float(
        trap_result["u_effective_support"] / (trap_result["v_effective_support"] + 1e-12)
    )
    trap_result.update(watcher.assess_trap_diffuseness(trap_result))

    trap_result["left_overlaps"] = left_overlaps
    trap_result["right_overlaps"] = right_overlaps
    trap_result["u_trap"] = u_trap
    trap_result["v_trap"] = v_trap
    trap_result["T_orig"] = T_orig
    trap_result["perm_evals_sorted"] = np.array(ww_layer.evals).copy()

    if params[wwcore.PLOT]:
        watcher.plot_trap_analysis(ww_layer, trap_result, params=params)

    trap_result.pop("left_overlaps", None)
    trap_result.pop("right_overlaps", None)
    trap_result.pop("u_trap", None)
    trap_result.pop("v_trap", None)
    if not params.get("_keep_trap_matrix", False):
        trap_result.pop("T_orig", None)
    trap_result.pop("perm_evals_sorted", None)

    return trap_result


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
