import hashlib
import numpy as np
import pandas as pd


def permutation_signature(indices):
    arr = np.asarray(indices, dtype=np.int64).ravel()
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def make_trap_identity_key(layer_id, seed, trap_index, n_traps, perm_signature):
    seed_str = "none" if seed is None else str(seed)
    perm_short = (perm_signature or "")[:16]
    return f"layer={layer_id}|seed={seed_str}|trap_index={trap_index}|n_traps={n_traps}|perm={perm_short}"


def abs_cosine(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return float(np.nan)
    a = np.asarray(vec_a, dtype=float).ravel()
    b = np.asarray(vec_b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return float(np.nan)
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if (not np.isfinite(na)) or (not np.isfinite(nb)) or na <= 0 or nb <= 0:
        return float(np.nan)
    return float(abs(np.dot(a, b) / (na * nb)))


def compare_numeric(a, b, rtol=1e-4, atol=1e-6):
    try:
        fa = float(a)
        fb = float(b)
    except Exception:
        return False
    if not np.isfinite(fa) or not np.isfinite(fb):
        return False
    return bool(np.isclose(fa, fb, rtol=rtol, atol=atol))


def verify_trap_rows(analyze_row, remove_row, rtol=1e-4, atol=1e-6, min_vector_cosine=0.999):
    perm_match = str(analyze_row.get("perm_signature", "")) == str(remove_row.get("perm_signature", ""))
    eval_close = compare_numeric(analyze_row.get("eval_perm", np.nan), remove_row.get("eval_perm", np.nan), rtol=rtol, atol=atol)
    bulk_close = compare_numeric(analyze_row.get("mp_bulk_max", np.nan), remove_row.get("mp_bulk_max", np.nan), rtol=rtol, atol=atol)
    delta_close = compare_numeric(analyze_row.get("trap_delta", np.nan), remove_row.get("trap_delta", np.nan), rtol=rtol, atol=atol)
    q_close = compare_numeric(analyze_row.get("trap_q", np.nan), remove_row.get("trap_q", np.nan), rtol=rtol, atol=atol)
    overlap_close = compare_numeric(
        analyze_row.get("trap_top_sector_overlap", np.nan),
        remove_row.get("trap_top_sector_overlap", np.nan),
        rtol=rtol,
        atol=atol,
    )

    v_cos = abs_cosine(analyze_row.get("v_trap", None), remove_row.get("v_trap", None))
    vec_ok = True if not np.isfinite(v_cos) else (v_cos >= float(min_vector_cosine))

    return {
        "perm_match": perm_match,
        "eval_perm_close": eval_close,
        "mp_bulk_max_close": bulk_close,
        "trap_delta_close": delta_close,
        "trap_q_close": q_close,
        "trap_top_sector_overlap_close": overlap_close,
        "v_abs_cosine": v_cos,
        "vector_close": vec_ok,
        "trap_verified": bool(perm_match and eval_close and bulk_close and delta_close and q_close and overlap_close and vec_ok),
    }


def build_trap_verification_row(analyze_row, remove_row, verify_dict, removed=False, removal_error=None):
    row = {
        "layer_id": analyze_row.get("layer_id", remove_row.get("layer_id", np.nan)),
        "trap_index": analyze_row.get("trap_index", remove_row.get("trap_index", np.nan)),
        "trap_seed": analyze_row.get("trap_seed", remove_row.get("trap_seed", np.nan)),
        "n_traps_analyze": analyze_row.get("n_traps", np.nan),
        "n_traps_remove": remove_row.get("n_traps", np.nan),
        "perm_signature_analyze": analyze_row.get("perm_signature", None),
        "perm_signature_remove": remove_row.get("perm_signature", None),
        "perm_match": verify_dict.get("perm_match", False),
        "eval_perm_analyze": analyze_row.get("eval_perm", np.nan),
        "eval_perm_remove": remove_row.get("eval_perm", np.nan),
        "eval_perm_close": verify_dict.get("eval_perm_close", False),
        "mp_bulk_max_analyze": analyze_row.get("mp_bulk_max", np.nan),
        "mp_bulk_max_remove": remove_row.get("mp_bulk_max", np.nan),
        "mp_bulk_max_close": verify_dict.get("mp_bulk_max_close", False),
        "trap_delta_analyze": analyze_row.get("trap_delta", np.nan),
        "trap_delta_remove": remove_row.get("trap_delta", np.nan),
        "trap_delta_close": verify_dict.get("trap_delta_close", False),
        "trap_q_analyze": analyze_row.get("trap_q", np.nan),
        "trap_q_remove": remove_row.get("trap_q", np.nan),
        "trap_q_close": verify_dict.get("trap_q_close", False),
        "trap_top_sector_overlap_analyze": analyze_row.get("trap_top_sector_overlap", np.nan),
        "trap_top_sector_overlap_remove": remove_row.get("trap_top_sector_overlap", np.nan),
        "trap_top_sector_overlap_close": verify_dict.get("trap_top_sector_overlap_close", False),
        "v_abs_cosine": verify_dict.get("v_abs_cosine", np.nan),
        "trap_verified": verify_dict.get("trap_verified", False),
        "removed": bool(removed),
        "removal_error": removal_error,
    }
    return row


def coerce_traps_dataframe(traps):
    if traps is None:
        return None
    if isinstance(traps, pd.DataFrame):
        return traps.copy()
    if isinstance(traps, pd.Series):
        return pd.DataFrame([traps.to_dict()])
    return pd.DataFrame(traps)
