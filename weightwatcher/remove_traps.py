import logging
import numbers
import hashlib
import copy
import numpy as np
import pandas as pd

from .RMT_Util import svd_full, unpermute_matrix
from .constants import DEFAULT_PARAMS, DEFAULT_START_ID, FAST_SVD, LAYER_TYPE, PEFT, PLOT, POOL, START_IDS, SVD_METHOD, DEFAULT_PEFT
from .constants import LAYERS
from .constants import WW_NAME
from .trap_histograms import plot_layer_trap_weight_histogram

logger = logging.getLogger(WW_NAME)


def _normalize_trap_rng(rng=None, seed=None):
    """Normalize trap permutation RNG to a reproducible numpy RandomState."""
    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both, for trap permutation reproducibility.")

    if rng is None and seed is None:
        return None

    if isinstance(rng, np.random.RandomState):
        return rng

    if isinstance(rng, np.random.Generator):
        raise ValueError(
            "Trap reproducibility requires numpy.random.RandomState or an int seed; "
            "numpy.random.Generator (e.g., np.random.default_rng()) is not supported."
        )

    if isinstance(rng, numbers.Integral):
        return np.random.RandomState(int(rng))

    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(int(seed))

    raise ValueError("rng/seed must be None, an int seed, or a numpy.random.RandomState.")


def apply_trap_mp_fit(ww, ww_layer, params=None):
    if params is None:
        params = DEFAULT_PARAMS.copy()
    ww.apply_esd(ww_layer, params)
    ww.apply_mp_fit(ww_layer, random=False, params=params)
    return ww_layer


def identify_trap_mode_indices(ww, ww_layer, svals=None, evals_desc=None):
    if evals_desc is None:
        if svals is None:
            W = ww_layer.Wmats[0]
            _, svals, _ = svd_full(W)
        evals_desc = np.asarray(svals, dtype=float) * np.asarray(svals, dtype=float)

    Q = ww_layer.N / ww_layer.M
    M = ww_layer.M
    sigma_mp = ww_layer.sigma_mp
    Wscale = ww_layer.W_scale

    bulk_max = (sigma_mp * (1 + 1 / np.sqrt(Q))) ** 2
    TW = 1 / np.sqrt(Q) * np.power(bulk_max, 2 / 3) * np.power(M, -2 / 3)
    bulk_max_TW = bulk_max + np.sqrt(TW)
    threshold = bulk_max_TW / (Wscale * Wscale)

    trap_mode_indices = np.where(evals_desc > threshold)[0]
    return trap_mode_indices.tolist()


def analyze_single_trap(ww, ww_layer, trap_mode_index, U_perm=None, S_perm=None, Vh_perm=None):
    def _top_percent_abs_mass(mat, percent):
        flat = np.abs(np.asarray(mat, dtype=float)).ravel()
        if flat.size == 0:
            return 0.0
        total = float(np.sum(flat))
        if total <= 0.0:
            return 0.0
        k = int(np.ceil((float(percent) / 100.0) * flat.size))
        k = max(1, min(k, flat.size))
        top_sum = float(np.sum(np.partition(flat, -k)[-k:]))
        return top_sum / total

    if U_perm is None or S_perm is None or Vh_perm is None:
        W_perm = ww_layer.Wmats[0]
        U_perm, S_perm, Vh_perm = svd_full(W_perm)

    sigma_perm = S_perm[trap_mode_index]
    u_trap = U_perm[:, trap_mode_index]
    v_trap = Vh_perm[trap_mode_index, :]
    T_perm = sigma_perm * np.outer(u_trap, v_trap)
    T_orig_norm = unpermute_matrix(T_perm, ww_layer.permute_ids[0])
    U_orig, _, Vh_orig = svd_full(T_orig_norm)
    top_5_mass = _top_percent_abs_mass(T_orig_norm, 5.0)
    top_10_mass = _top_percent_abs_mass(T_orig_norm, 10.0)

    return {
        "trap_mode_index": trap_mode_index,
        "sigma_perm": sigma_perm,
        "u_trap_perm": u_trap,
        "v_trap_perm": v_trap,
        "u_trap": U_orig[:, 0],
        "v_trap": Vh_orig[0, :],
        "T_perm": T_perm,
        "T_orig_norm": T_orig_norm,
        "top_5_mass": float(top_5_mass),
        "top_10_mass": float(top_10_mass),
    }


def collect_trap_artifacts(ww, ww_layer, params=None, seed=None, rng=None, already_randomized=False, permuted_ids=None, return_state=False):
    if params is None:
        params = DEFAULT_PARAMS.copy()

    if rng is None and seed is None and isinstance(params, dict):
        seed = params.get("seed", None)
    rng = _normalize_trap_rng(rng=rng, seed=seed)

    analysis_layer = ww_layer.copy()
    analysis_layer.Wmats = [ww_layer.Wmats[0].copy()]
    analysis_layer.w_norm = 1.0
    analysis_layer.permute_ids = []

    ww.apply_normalize_Wmats(analysis_layer, params)
    if already_randomized:
        if permuted_ids is not None:
            analysis_layer.permute_ids = [np.asarray(permuted_ids).astype(int)]
    elif permuted_ids is not None:
        pids = np.asarray(permuted_ids).astype(int)
        analysis_layer.permute_ids = [pids]
        analysis_layer.Wmats = [analysis_layer.Wmats[0].flatten()[pids].reshape(analysis_layer.Wmats[0].shape)]
    else:
        ww.apply_permute_W(analysis_layer, params, rng=rng)
    permute_fingerprint = None
    if len(analysis_layer.permute_ids) > 0:
        perm_arr = np.asarray(analysis_layer.permute_ids[0])
        permute_fingerprint = hashlib.sha1(perm_arr.tobytes()).hexdigest()
    apply_trap_mp_fit(ww, analysis_layer, params)
    W_perm = analysis_layer.Wmats[0]
    U_perm, S_perm, Vh_perm = svd_full(W_perm)
    trap_mode_indices = identify_trap_mode_indices(ww, analysis_layer, svals=S_perm)

    artifacts = []
    for i, trap_mode_index in enumerate(trap_mode_indices, start=1):
        artifact = analyze_single_trap(ww, analysis_layer, trap_mode_index, U_perm=U_perm, S_perm=S_perm, Vh_perm=Vh_perm)
        artifact["trap_index"] = i
        artifact["T_orig_raw"] = artifact["T_orig_norm"] / analysis_layer.w_norm
        artifact["permute_fingerprint"] = permute_fingerprint
        artifacts.append(artifact)

    if return_state:
        state = {
            "layer_id": int(getattr(ww_layer, "layer_id", -1)),
            "permuted_ids": np.asarray(analysis_layer.permute_ids[0]).copy() if analysis_layer.permute_ids else None,
            "permute_fingerprint": permute_fingerprint,
            "U_perm": U_perm, "S_perm": S_perm, "Vh_perm": Vh_perm,
            "trap_mode_indices": list(trap_mode_indices),
            "artifacts": artifacts,
            "already_randomized": bool(already_randomized),
        }
        return artifacts, state
    return artifacts


def make_stat_matched_random_matrix(T, rng):
    G = rng.standard_normal(T.shape)
    G = G - np.mean(G)
    g_std = np.std(G)
    if g_std > 0:
        G = G / g_std
    else:
        G = np.zeros_like(T)
    return np.mean(T) + np.std(T) * G


def apply_remove_traps(ww, ww_layer, trap_indices, params=None, seed=None, rng=None, trap_artifacts=None, trap_state=None):
    if params is None:
        params = DEFAULT_PARAMS.copy()
    if trap_indices is None or len(trap_indices) == 0:
        raise ValueError("trap_indices must be a non-empty list of 1-based indices")

    if ww_layer.the_type != LAYER_TYPE.DENSE or len(ww_layer.Wmats) != 1 or ww_layer.Wmats[0].ndim != 2:
        raise NotImplementedError("remove_traps currently supports single 2D dense matrices only")

    requested = sorted(set(trap_indices))
    if any(idx < 1 for idx in requested):
        raise ValueError("trap indices are 1-based and must be >= 1")

    layer_seed = seed
    if layer_seed is None and isinstance(params, dict):
        layer_seed = params.get("seed", None)

    # If an RNG object is already provided, do not also pass seed into
    # normalization (that helper enforces exactly one of rng/seed).
    permute_rng = _normalize_trap_rng(rng=rng, seed=None if rng is not None else layer_seed)

    permute_seed = layer_seed
    replacement_seed = None if layer_seed is None else layer_seed + 1
    replacement_rng = np.random.default_rng(replacement_seed)

    artifacts = trap_artifacts
    if artifacts is None and isinstance(trap_state, dict):
        artifacts = trap_state.get("artifacts")
    if artifacts is None:
        artifacts = collect_trap_artifacts(
            ww,
            ww_layer,
            params=params,
            seed=None if permute_rng is not None else permute_seed,
            rng=permute_rng,
            already_randomized=bool(isinstance(trap_state, dict) and trap_state.get("already_randomized", False)),
            permuted_ids=(trap_state.get("permuted_ids") if isinstance(trap_state, dict) else None),
        )
    valid_indices = [idx for idx in requested if idx <= len(artifacts)]
    if len(valid_indices) < len(requested):
        logger.warning(
            f"Skipping invalid trap indices {set(requested) - set(valid_indices)}; "
            f"only {len(artifacts)} traps detected"
        )
    if len(valid_indices) == 0:
        logger.warning("No valid traps to remove for this layer; skipping")
        return ww_layer
    requested = valid_indices

    if params.get(PLOT, False):
        max_sigma = max(float(a.get("sigma_perm", 0.0)) for a in artifacts) if len(artifacts) > 0 else 0.0
        trap_infos = []
        for idx in requested:
            artifact = artifacts[idx - 1]
            rel_sigma = float(artifact.get("sigma_perm", 0.0)) / (max_sigma + 1e-12)
            if rel_sigma >= 0.8:
                assessment = "localized_risky"
            elif rel_sigma <= 0.35:
                assessment = "benign_diffuse"
            else:
                assessment = "mixed"
            trap_infos.append(
                {
                    "trap_index": idx,
                    "trap_matrix": artifact["T_orig_raw"],
                    "trap_assessment": assessment,
                    "trap_risk_score": rel_sigma,
                }
            )

        plot_layer_trap_weight_histogram(
            ww_layer,
            trap_infos,
            params=params,
            method_tag="remove_traps",
        )

    old_W = ww_layer.Wmats[0]
    new_W = old_W.copy()
    for idx in requested:
        T_orig_raw = artifacts[idx - 1]["T_orig_raw"]
        R_orig = make_stat_matched_random_matrix(T_orig_raw, replacement_rng)
        new_W = new_W - T_orig_raw + R_orig

    ww.replace_layer_weights(ww_layer.layer_id, ww_layer.framework_layer, new_W)
    ww_layer.Wmats = [new_W]
    return ww_layer


def _trap_indices_from_traps_df(traps):
    """Extract unique 1-based trap indices from a traps DataFrame-like input."""
    if traps is None:
        return None
    if isinstance(traps, pd.DataFrame):
        trap_df = traps
    else:
        trap_df = pd.DataFrame(traps)
    if "trap_index" not in trap_df.columns:
        raise ValueError("traps must include a 'trap_index' column")
    indices = trap_df["trap_index"].dropna().astype(int).tolist()
    indices = sorted(set(indices))
    if len(indices) == 0:
        raise ValueError("traps did not contain any valid trap_index values")
    return indices


def remove_traps(ww, model=None, layers=[], trap_indices=None, traps=None, seed=None, rng=None, pool=True, plot=True,
                 verify_traps=False, return_analyze=False, start_ids=DEFAULT_START_ID, svd_method=FAST_SVD,
                 base_model=None, peft=DEFAULT_PEFT, trap_artifacts=None, trap_state=None, already_randomized=False):
    # PR359 compatibility path: passing traps=<DataFrame> instead of trap_indices=[...]
    if trap_indices is None and traps is not None:
        trap_indices = _trap_indices_from_traps_df(traps)

    if trap_indices is None or len(trap_indices) == 0:
        raise ValueError("trap_indices must be provided and non-empty (or pass traps with trap_index column)")

    ww.set_model_(model)
    params = DEFAULT_PARAMS.copy()
    params[POOL] = pool
    params[LAYERS] = layers
    params[PLOT] = plot
    params[START_IDS] = start_ids
    params[SVD_METHOD] = svd_method
    params[PEFT] = peft
    params["seed"] = seed
    params["rng"] = _normalize_trap_rng(rng=rng, seed=seed)
    params["already_randomized"] = bool(already_randomized)

    if not ww.__class__.valid_params(params):
        raise Exception(f"Error, params not valid: \n {params}")
    params = ww.normalize_params(params)

    layer_iterator = ww.make_layer_iterator(model=ww.model, layers=layers, params=params, base_model=base_model)
    verify_rows = []
    traps_df = pd.DataFrame(traps) if traps is not None else None
    for ww_layer in layer_iterator:
        if not ww_layer.skipped and ww_layer.has_weights:
            layer_traps = None
            if traps_df is not None and "layer_id" in traps_df.columns:
                layer_traps = traps_df[traps_df["layer_id"].astype(int) == int(ww_layer.layer_id)].copy()
                if len(layer_traps) > 0:
                    if "trap_index" in layer_traps.columns:
                        trap_indices = sorted(set(layer_traps["trap_index"].dropna().astype(int).tolist()))
                    else:
                        raise ValueError("traps DataFrame must include trap_index")

            pre_artifacts = trap_artifacts
            if pre_artifacts is None and isinstance(trap_state, dict):
                layer_state = trap_state.get("layers", {}).get(int(ww_layer.layer_id), {})
                pre_artifacts = layer_state.get("artifacts")
            if pre_artifacts is None:
                pre_artifacts = collect_trap_artifacts(
                    ww,
                    ww_layer,
                    params=params,
                    seed=None if params["rng"] is not None else seed,
                    rng=params["rng"],
                    already_randomized=bool(already_randomized),
                    permuted_ids=(trap_state.get("permuted_ids", {}).get(int(ww_layer.layer_id)) if isinstance(trap_state, dict) else None),
                )
            pre_by_index = {int(a["trap_index"]): a for a in pre_artifacts}
            identity_ok = True
            identity_reason = "ok"
            if layer_traps is not None and len(layer_traps) > 0:
                for _, trow in layer_traps.iterrows():
                    tidx = int(trow["trap_index"])
                    art = pre_by_index.get(tidx)
                    if art is None:
                        identity_ok = False
                        identity_reason = f"trap_index_{tidx}_missing"
                        break
                    if "trap_mode_index" in layer_traps.columns and not pd.isna(trow.get("trap_mode_index")):
                        if int(trow["trap_mode_index"]) != int(art["trap_mode_index"]):
                            identity_ok = False
                            identity_reason = f"trap_mode_mismatch_{tidx}"
                            break
                    if "permute_fingerprint" in layer_traps.columns and pd.notna(trow.get("permute_fingerprint")):
                        if str(trow["permute_fingerprint"]) != str(art.get("permute_fingerprint")):
                            identity_ok = False
                            identity_reason = f"permute_mismatch_{tidx}"
                            break

            if not identity_ok:
                raise ValueError(f"Trap identity verification failed for layer {ww_layer.layer_id}: {identity_reason}")

            apply_remove_traps(ww, ww_layer, trap_indices=trap_indices, params=params, seed=seed, rng=params["rng"], trap_artifacts=pre_artifacts, trap_state=(trap_state.get("layers", {}).get(int(ww_layer.layer_id), {}) if isinstance(trap_state, dict) else None))
            if verify_traps:
                remaining = collect_trap_artifacts(
                    ww,
                    ww_layer,
                    params=params,
                    seed=None if params["rng"] is not None else seed,
                    rng=params["rng"],
                )
                verify_rows.append(
                    {
                        "layer_id": int(ww_layer.layer_id),
                        "requested_trap_indices": list(trap_indices),
                        "identity_verified": bool(identity_ok),
                        "identity_reason": identity_reason,
                        "remaining_traps": len(remaining),
                        "verify_passed": len(remaining) == 0,
                    }
                )

    if return_analyze:
        verify_df = pd.DataFrame.from_records(verify_rows)
        return model, verify_df
    return model


def randomize_model(ww, model=None, layers=None, pool=True, start_ids=DEFAULT_START_ID, svd_method=FAST_SVD,
                    base_model=None, peft=DEFAULT_PEFT, rng=None, return_state=False):
    """Return a randomized copy of model and per-layer permutation ids."""
    if layers is None:
        layers = []

    randomized_model = copy.deepcopy(model if model is not None else ww.model)
    ww.set_model_(randomized_model)

    params = DEFAULT_PARAMS.copy()
    params[POOL] = pool
    params[LAYERS] = layers
    params[START_IDS] = start_ids
    params[SVD_METHOD] = svd_method
    params[PEFT] = peft
    params["rng"] = _normalize_trap_rng(rng=rng)

    params = ww.normalize_params(params)
    layer_iterator = ww.make_layer_iterator(model=ww.model, layers=layers, params=params, base_model=base_model)

    permuted_ids = {}
    layer_states = {}
    for ww_layer in layer_iterator:
        if not ww_layer.skipped and ww_layer.has_weights:
            ww.apply_normalize_Wmats(ww_layer, params)
            ww.apply_permute_W(ww_layer, params, rng=params["rng"])
            if ww_layer.permute_ids:
                pid = np.asarray(ww_layer.permute_ids[0]).copy()
                lid = int(ww_layer.layer_id)
                permuted_ids[lid] = pid
                layer_states[lid] = {"layer_id": lid, "permuted_ids": pid, "permute_fingerprint": hashlib.sha1(pid.tobytes()).hexdigest(), "already_randomized": True}
            ww.replace_layer_weights(ww_layer.layer_id, ww_layer.framework_layer, ww_layer.Wmats[0])

    if return_state:
        return randomized_model, {"permuted_ids": permuted_ids, "layers": layer_states, "already_randomized": True}
    return randomized_model, permuted_ids
