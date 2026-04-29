import logging
import numbers
import numpy as np

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


def identify_bulk_mode_indices(ww, ww_layer):
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

    bulk_mode_indices = np.where(evals_desc <= threshold)[0]
    return bulk_mode_indices.tolist()


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

    artifacts = []
    for i, trap_mode_index in enumerate(trap_mode_indices, start=1):
        artifact = analyze_single_trap(ww, analysis_layer, trap_mode_index)
        artifact["trap_index"] = i
        artifact["T_orig_raw"] = artifact["T_orig_norm"] / analysis_layer.w_norm
        artifacts.append(artifact)

    return artifacts


def collect_bulk_artifacts(ww, ww_layer, params=None, seed=None, rng=None):
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
    bulk_mode_indices = identify_bulk_mode_indices(ww, analysis_layer)

    artifacts = []
    for i, bulk_mode_index in enumerate(bulk_mode_indices, start=1):
        artifact = analyze_single_trap(ww, analysis_layer, bulk_mode_index)
        artifact["bulk_index"] = i
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


def apply_remove_traps(
    ww,
    ww_layer,
    trap_indices=None,
    bulk_indices=None,
    num_random_bulk_vectors=0,
    params=None,
    seed=None,
    rng=None,
):
    if params is None:
        params = DEFAULT_PARAMS.copy()
    if (trap_indices is None or len(trap_indices) == 0) and (bulk_indices is None or len(bulk_indices) == 0) and int(num_random_bulk_vectors) <= 0:
        raise ValueError("Specify at least one of trap_indices, bulk_indices, or num_random_bulk_vectors > 0")

    if ww_layer.the_type != LAYER_TYPE.DENSE or len(ww_layer.Wmats) != 1 or ww_layer.Wmats[0].ndim != 2:
        raise NotImplementedError("remove_traps currently supports single 2D dense matrices only")

    requested = sorted(set(trap_indices or []))
    if any(idx < 1 for idx in requested):
        raise ValueError("trap indices are 1-based and must be >= 1")
    requested_bulk = sorted(set(bulk_indices or []))
    if any(idx < 1 for idx in requested_bulk):
        raise ValueError("bulk indices are 1-based and must be >= 1")
    if int(num_random_bulk_vectors) < 0:
        raise ValueError("num_random_bulk_vectors must be >= 0")

    layer_seed = seed
    if layer_seed is None and isinstance(params, dict):
        layer_seed = params.get("seed", None)

    # If an RNG object is already provided, do not also pass seed into
    # normalization (that helper enforces exactly one of rng/seed).
    permute_rng = _normalize_trap_rng(rng=rng, seed=None if rng is not None else layer_seed)

    permute_seed = layer_seed
    replacement_seed = None if layer_seed is None else layer_seed + 1
    replacement_rng = np.random.default_rng(replacement_seed)

    trap_artifacts = collect_trap_artifacts(
        ww,
        ww_layer,
        params=params,
        seed=None if permute_rng is not None else permute_seed,
        rng=permute_rng,
    )
    valid_indices = [idx for idx in requested if idx <= len(trap_artifacts)]
    if len(valid_indices) < len(requested):
        logger.warning(
            f"Skipping invalid trap indices {set(requested) - set(valid_indices)}; "
            f"only {len(trap_artifacts)} traps detected"
        )
    requested = valid_indices

    bulk_artifacts = collect_bulk_artifacts(
        ww,
        ww_layer,
        params=params,
        seed=None if permute_rng is not None else permute_seed,
        rng=permute_rng,
    )
    valid_bulk_indices = [idx for idx in requested_bulk if idx <= len(bulk_artifacts)]
    if len(valid_bulk_indices) < len(requested_bulk):
        logger.warning(
            f"Skipping invalid bulk indices {set(requested_bulk) - set(valid_bulk_indices)}; "
            f"only {len(bulk_artifacts)} bulk vectors available"
        )
    requested_bulk = valid_bulk_indices

    random_bulk_count = int(num_random_bulk_vectors)
    if random_bulk_count > 0:
        available_bulk = [idx for idx in range(1, len(bulk_artifacts) + 1) if idx not in requested_bulk]
        if len(available_bulk) == 0:
            logger.warning("Requested random bulk vector removal, but no bulk vectors are available")
        else:
            if random_bulk_count > len(available_bulk):
                logger.warning(
                    "Requested %d random bulk vectors but only %d are available; selecting all available",
                    random_bulk_count,
                    len(available_bulk),
                )
                random_bulk_count = len(available_bulk)
            random_choices = replacement_rng.choice(available_bulk, size=random_bulk_count, replace=False).tolist()
            requested_bulk = sorted(set(requested_bulk + random_choices))

    if len(requested) == 0 and len(requested_bulk) == 0:
        logger.warning("No valid trap or bulk vectors to remove for this layer; skipping")
        return ww_layer

    if params.get(PLOT, False):
        max_sigma = max(float(a.get("sigma_perm", 0.0)) for a in trap_artifacts) if len(trap_artifacts) > 0 else 0.0
        trap_infos = []
        for idx in requested:
            artifact = trap_artifacts[idx - 1]
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
        T_orig_raw = trap_artifacts[idx - 1]["T_orig_raw"]
        R_orig = make_stat_matched_random_matrix(T_orig_raw, replacement_rng)
        new_W = new_W - T_orig_raw + R_orig

    for idx in requested_bulk:
        T_orig_raw = bulk_artifacts[idx - 1]["T_orig_raw"]
        R_orig = make_stat_matched_random_matrix(T_orig_raw, replacement_rng)
        new_W = new_W - T_orig_raw + R_orig

    ww.replace_layer_weights(ww_layer.layer_id, ww_layer.framework_layer, new_W)
    ww_layer.Wmats = [new_W]
    return ww_layer


def remove_traps(ww, model=None, layers=[], trap_indices=None, bulk_indices=None, num_random_bulk_vectors=0,
                 seed=None, rng=None, pool=True, plot=True,
                 start_ids=DEFAULT_START_ID, svd_method=FAST_SVD, base_model=None, peft=DEFAULT_PEFT):
    if (trap_indices is None or len(trap_indices) == 0) and (bulk_indices is None or len(bulk_indices) == 0) and int(num_random_bulk_vectors) <= 0:
        raise ValueError("Specify at least one of trap_indices, bulk_indices, or num_random_bulk_vectors > 0")

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

    layer_iterator = ww.make_layer_iterator(model=ww.model, layers=layers, params=params, base_model=base_model)
    for ww_layer in layer_iterator:
        if not ww_layer.skipped and ww_layer.has_weights:
            apply_remove_traps(
                ww,
                ww_layer,
                trap_indices=trap_indices,
                bulk_indices=bulk_indices,
                num_random_bulk_vectors=num_random_bulk_vectors,
                params=params,
                seed=seed,
                rng=params["rng"],
            )

    return model
