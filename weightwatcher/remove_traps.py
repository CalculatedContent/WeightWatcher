import logging
import numbers
import numpy as np
import pandas as pd

from .RMT_Util import svd_full, unpermute_matrix
from .constants import DEFAULT_PARAMS, DEFAULT_START_ID, FAST_SVD, LAYER_TYPE, PEFT, PLOT, POOL, START_IDS, SVD_METHOD, DEFAULT_PEFT
from .constants import LAYERS
from .constants import WW_NAME
from .trap_histograms import plot_layer_trap_weight_histogram
from . import trap_identity

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


def identify_trap_mode_indices(ww, ww_layer):
    W = ww_layer.Wmats[0]
    _, svals, _ = svd_full(W)
    evals_desc = svals * svals

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


def analyze_single_trap(ww, ww_layer, trap_mode_index):
    W_perm = ww_layer.Wmats[0]
    U_perm, S_perm, Vh_perm = svd_full(W_perm)

    sigma_perm = S_perm[trap_mode_index]
    u_trap = U_perm[:, trap_mode_index]
    v_trap = Vh_perm[trap_mode_index, :]
    T_perm = sigma_perm * np.outer(u_trap, v_trap)
    T_orig_norm = unpermute_matrix(T_perm, ww_layer.permute_ids[0])
    U_orig, _, Vh_orig = svd_full(T_orig_norm)

    return {
        "trap_mode_index": trap_mode_index,
        "sigma_perm": sigma_perm,
        "u_trap_perm": u_trap,
        "v_trap_perm": v_trap,
        "u_trap": U_orig[:, 0],
        "v_trap": Vh_orig[0, :],
        "T_perm": T_perm,
        "T_orig_norm": T_orig_norm,
    }


def collect_trap_artifacts(ww, ww_layer, params=None, seed=None, rng=None):
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
    ww.apply_permute_W(analysis_layer, params, rng=rng)
    apply_trap_mp_fit(ww, analysis_layer, params)
    trap_mode_indices = identify_trap_mode_indices(ww, analysis_layer)
    n_traps = int(len(trap_mode_indices))
    perm_ids = analysis_layer.permute_ids[0] if len(analysis_layer.permute_ids) > 0 else np.array([], dtype=int)
    perm_sig = trap_identity.permutation_signature(perm_ids)

    artifacts = []
    for i, trap_mode_index in enumerate(trap_mode_indices, start=1):
        artifact = analyze_single_trap(ww, analysis_layer, trap_mode_index)
        artifact["trap_index"] = i
        artifact["trap_seed"] = seed
        artifact["n_traps"] = n_traps
        artifact["perm_signature"] = perm_sig
        artifact["permutation_n"] = int(len(np.asarray(perm_ids).ravel()))
        artifact["permutation_mode"] = "index_permutation"
        artifact["trap_identity_key"] = trap_identity.make_trap_identity_key(
            layer_id=analysis_layer.layer_id,
            seed=seed,
            trap_index=i - 1,
            n_traps=n_traps,
            perm_signature=perm_sig,
        )
        artifact["layer_id"] = analysis_layer.layer_id
        artifact["eval_perm"] = float(artifact["sigma_perm"] ** 2)
        artifact["mp_bulk_max"] = float(getattr(analysis_layer, "bulk_max", np.nan))
        mp_bulk_max = float(getattr(analysis_layer, "bulk_max", np.nan))
        if np.isfinite(mp_bulk_max) and mp_bulk_max > 0:
            artifact["trap_delta"] = float(max(artifact["eval_perm"] - mp_bulk_max, 0.0) / mp_bulk_max)
        else:
            artifact["trap_delta"] = float(np.nan)
        artifact["T_orig_raw"] = artifact["T_orig_norm"] / analysis_layer.w_norm
        artifacts.append(artifact)

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


def apply_remove_traps(ww, ww_layer, trap_indices, params=None, seed=None, rng=None):
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

    artifacts = collect_trap_artifacts(
        ww,
        ww_layer,
        params=params,
        seed=None if permute_rng is not None else permute_seed,
        rng=permute_rng,
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


def remove_traps(
    ww,
    model=None,
    layers=[],
    trap_indices=None,
    seed=None,
    rng=None,
    pool=True,
    plot=True,
    start_ids=DEFAULT_START_ID,
    svd_method=FAST_SVD,
    base_model=None,
    peft=DEFAULT_PEFT,
    verify_traps=False,
    return_analyze=False,
    traps=None,
    rtol=1e-4,
    atol=1e-6,
    min_vector_cosine=0.999,
):
    traps_df = trap_identity.coerce_traps_dataframe(traps)
    if traps_df is not None and len(traps_df) > 0 and (trap_indices is None or len(trap_indices) == 0):
        trap_indices = [int(i) + 1 for i in traps_df["trap_index"].astype(int).tolist()]

    if trap_indices is None or len(trap_indices) == 0:
        raise ValueError("trap_indices must be provided and non-empty")

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

    if not ww.__class__.valid_params(params):
        raise Exception(f"Error, params not valid: \n {params}")
    params = ww.normalize_params(params)

    analyze_df = None
    remove_rows = []
    needs_verify = bool(verify_traps or return_analyze or traps_df is not None)
    if needs_verify:
        analyze_df = ww.analyze_traps(
            model=model,
            layers=layers,
            plot=False,
            savefig=False,
            pool=pool,
            start_ids=start_ids,
            peft=peft,
            seed=seed,
            rng=None,
            return_burden_raw=True,
        )

    layer_iterator = ww.make_layer_iterator(model=ww.model, layers=layers, params=params, base_model=base_model)
    for ww_layer in layer_iterator:
        if not ww_layer.skipped and ww_layer.has_weights:
            layer_artifacts = None
            if needs_verify:
                layer_artifacts = collect_trap_artifacts(
                    ww,
                    ww_layer,
                    params=params,
                    seed=seed,
                    rng=None,
                )

            layer_indices = sorted(set([int(i) for i in trap_indices]))
            removed_flag = True
            removal_error = None
            if needs_verify:
                for idx in layer_indices:
                    if layer_artifacts is None or idx < 1 or idx > len(layer_artifacts):
                        continue
                    remove_row = layer_artifacts[idx - 1]
                    if traps_df is not None and len(traps_df) > 0:
                        candidates = traps_df[
                            (traps_df["layer_id"].astype(int) == int(ww_layer.layer_id))
                            & (traps_df["trap_index"].astype(int) == int(idx - 1))
                        ]
                        if len(candidates) == 0:
                            analyze_row = pd.Series(dtype=float)
                        else:
                            analyze_row = candidates.iloc[0]
                    else:
                        candidates = analyze_df[
                            (analyze_df["layer_id"].astype(int) == int(ww_layer.layer_id))
                            & (analyze_df["trap_index"].astype(int) == int(idx - 1))
                        ]
                        analyze_row = candidates.iloc[0] if len(candidates) > 0 else pd.Series(dtype=float)
                    if "trap_q" not in remove_row:
                        remove_row["trap_q"] = analyze_row.get("trap_q", np.nan)
                    if "trap_top_sector_overlap" not in remove_row:
                        remove_row["trap_top_sector_overlap"] = analyze_row.get("trap_top_sector_overlap", np.nan)
                    if "v_trap" not in remove_row:
                        remove_row["v_trap"] = analyze_row.get("v_trap", None)

                    verify = trap_identity.verify_trap_rows(
                        analyze_row,
                        remove_row,
                        rtol=rtol,
                        atol=atol,
                        min_vector_cosine=min_vector_cosine,
                    )
                    vrow = trap_identity.build_trap_verification_row(
                        analyze_row=analyze_row,
                        remove_row=remove_row,
                        verify_dict=verify,
                        removed=False,
                        removal_error=None,
                    )
                    remove_rows.append(vrow)
                    if verify_traps and (not verify.get("trap_verified", False)):
                        raise RuntimeError(
                            f"Trap verification failed for layer {ww_layer.layer_id}, trap_index={idx - 1}"
                        )

            apply_remove_traps(ww, ww_layer, trap_indices=trap_indices, params=params, seed=seed, rng=params["rng"])
            if needs_verify and len(remove_rows) > 0:
                for i in range(len(remove_rows)):
                    if int(remove_rows[i].get("layer_id", -1)) == int(ww_layer.layer_id):
                        remove_rows[i]["removed"] = removed_flag
                        remove_rows[i]["removal_error"] = removal_error
    if needs_verify:
        remove_meta_df = pd.DataFrame(remove_rows)
        return model, remove_meta_df
    return model
