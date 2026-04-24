import copy
import math
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def safe_float(x):
    try:
        val = float(x)
    except Exception:
        return float(np.nan)
    if not np.isfinite(val):
        return float(np.nan)
    return float(val)


def vector_ipr(vec):
    v = np.asarray(vec, dtype=float).ravel()
    if v.size == 0:
        return float(np.nan)
    norm = np.linalg.norm(v)
    if (not np.isfinite(norm)) or norm <= 0.0:
        return float(np.nan)
    v = v / norm
    return float(np.sum(np.abs(v) ** 4))


def localization_uniform_centered(vec, clip=True):
    ipr = vector_ipr(vec)
    if not np.isfinite(ipr):
        return float(np.nan), float(np.nan)
    n = len(np.asarray(vec).ravel())
    if n <= 1:
        q = 1.0
    else:
        q = (n * ipr - 1.0) / (n - 1.0)
        if clip:
            q = float(np.clip(q, 0.0, 1.0))
    return float(ipr), float(q)


def localization_porter_thomas_centered(vec, beta="real", clip=True):
    ipr = vector_ipr(vec)
    if not np.isfinite(ipr):
        return float(np.nan), float(np.nan)
    n = len(np.asarray(vec).ravel())
    if n <= 1:
        return float(ipr), 1.0

    if beta == "complex":
        expected_ipr = 2.0 / (n + 1.0)
    else:
        expected_ipr = 3.0 / (n + 2.0)

    denom = 1.0 - expected_ipr
    if denom <= 0:
        return float(ipr), float(np.nan)

    q_pt = (ipr - expected_ipr) / denom
    if clip:
        q_pt = float(np.clip(q_pt, 0.0, 1.0))
    return float(ipr), float(q_pt)


def spectral_excess(eval_perm, mp_bulk_max, perm_total_variance, mode="edge_ratio_current"):
    eval_perm = safe_float(eval_perm)
    mp_bulk_max = safe_float(mp_bulk_max)
    total_var = safe_float(perm_total_variance)
    if not np.isfinite(eval_perm):
        return float(np.nan)

    raw_excess = float(np.nan)
    if np.isfinite(mp_bulk_max):
        raw_excess = float(max(eval_perm - mp_bulk_max, 0.0))

    if mode in {"edge_ratio_current", "edge_ratio_linear"}:
        if (not np.isfinite(mp_bulk_max)) or mp_bulk_max <= 0.0:
            return float(np.nan)
        return float(raw_excess / mp_bulk_max)
    if mode == "total_excess":
        if (not np.isfinite(total_var)) or total_var <= 0.0 or (not np.isfinite(raw_excess)):
            return float(np.nan)
        return float(raw_excess / total_var)
    if mode == "total_fraction":
        if (not np.isfinite(total_var)) or total_var <= 0.0:
            return float(np.nan)
        return float(eval_perm / total_var)
    if mode == "raw_excess":
        return float(raw_excess)
    if mode == "log_edge_ratio":
        if (not np.isfinite(mp_bulk_max)) or mp_bulk_max <= 0.0 or eval_perm <= 0.0:
            return float(np.nan)
        return float(max(math.log(eval_perm / mp_bulk_max), 0.0))
    raise ValueError(f"Unknown spectral mode: {mode}")


def combine_lr(left, right, method):
    l = safe_float(left)
    r = safe_float(right)
    if method == "left":
        return l
    if method == "right":
        return r
    if (not np.isfinite(l)) or (not np.isfinite(r)):
        return float(np.nan)
    if method == "mean":
        return float(0.5 * (l + r))
    if method == "geom":
        return float(np.sqrt(max(l, 0.0) * max(r, 0.0)))
    if method == "min":
        return float(min(l, r))
    if method == "max":
        return float(max(l, r))
    if method == "product":
        return float(l * r)
    raise ValueError(f"Unknown left/right combine method: {method}")


def compute_top_sector_overlap_pair(left_overlaps, right_overlaps, top_sector_l):
    ell = int(top_sector_l)
    if ell < 1:
        raise ValueError("top_sector_l must be >= 1")
    left = np.asarray(left_overlaps, dtype=float).ravel()
    right = np.asarray(right_overlaps, dtype=float).ravel()

    if left.size == 0:
        left_sum = float(np.nan)
        ell_left = 0
    else:
        ell_left = min(ell, int(left.size))
        left_sum = float(np.sum(left[:ell_left]))

    if right.size == 0:
        right_sum = float(np.nan)
        ell_right = 0
    else:
        ell_right = min(ell, int(right.size))
        right_sum = float(np.sum(right[:ell_right]))

    return left_sum, right_sum, int(ell_left), int(ell_right)


def compute_burden_components(
    eval_perm,
    mp_bulk_max,
    perm_total_variance,
    u_perm,
    v_perm,
    u_trap,
    v_trap,
    left_overlaps,
    right_overlaps,
    top_sector_l=1,
    beta="real",
):
    out = {}
    out["trap_perm_total_variance"] = safe_float(perm_total_variance)
    out["trap_spectral_edge_ratio_current"] = spectral_excess(
        eval_perm, mp_bulk_max, perm_total_variance, mode="edge_ratio_current"
    )
    out["trap_spectral_total_excess"] = spectral_excess(
        eval_perm, mp_bulk_max, perm_total_variance, mode="total_excess"
    )
    out["trap_spectral_total_fraction"] = spectral_excess(
        eval_perm, mp_bulk_max, perm_total_variance, mode="total_fraction"
    )
    out["trap_spectral_raw_excess"] = spectral_excess(
        eval_perm, mp_bulk_max, perm_total_variance, mode="raw_excess"
    )
    out["trap_spectral_log_edge_ratio"] = spectral_excess(
        eval_perm, mp_bulk_max, perm_total_variance, mode="log_edge_ratio"
    )

    ipr_l_perm, q_l_perm = localization_uniform_centered(u_perm, clip=True)
    ipr_r_perm, q_r_perm = localization_uniform_centered(v_perm, clip=True)
    ipr_l_trap, q_l_trap = localization_uniform_centered(u_trap, clip=True)
    ipr_r_trap, q_r_trap = localization_uniform_centered(v_trap, clip=True)

    out["trap_ipr_left_perm"] = ipr_l_perm
    out["trap_ipr_right_perm"] = ipr_r_perm
    out["trap_ipr_left_trap"] = ipr_l_trap
    out["trap_ipr_right_trap"] = ipr_r_trap

    out["trap_q_uniform_left_perm"] = q_l_perm
    out["trap_q_uniform_right_perm"] = q_r_perm
    out["trap_q_uniform_left_trap"] = q_l_trap
    out["trap_q_uniform_right_trap"] = q_r_trap

    _, q_pt_l_perm = localization_porter_thomas_centered(u_perm, beta=beta, clip=True)
    _, q_pt_r_perm = localization_porter_thomas_centered(v_perm, beta=beta, clip=True)
    _, q_pt_l_trap = localization_porter_thomas_centered(u_trap, beta=beta, clip=True)
    _, q_pt_r_trap = localization_porter_thomas_centered(v_trap, beta=beta, clip=True)

    out["trap_q_pt_left_perm"] = q_pt_l_perm
    out["trap_q_pt_right_perm"] = q_pt_r_perm
    out["trap_q_pt_left_trap"] = q_pt_l_trap
    out["trap_q_pt_right_trap"] = q_pt_r_trap

    out["trap_q_pt_perm_lr_geom"] = combine_lr(q_pt_l_perm, q_pt_r_perm, "geom")
    out["trap_q_pt_perm_lr_min"] = combine_lr(q_pt_l_perm, q_pt_r_perm, "min")
    out["trap_q_pt_trap_lr_geom"] = combine_lr(q_pt_l_trap, q_pt_r_trap, "geom")
    out["trap_q_pt_trap_lr_min"] = combine_lr(q_pt_l_trap, q_pt_r_trap, "min")

    left_ov, right_ov, ell_l, ell_r = compute_top_sector_overlap_pair(
        left_overlaps, right_overlaps, top_sector_l=top_sector_l
    )
    out["top_sector_l_effective_left"] = ell_l
    out["top_sector_l_effective_right"] = ell_r
    out["trap_top_sector_overlap_left"] = left_ov
    out["trap_top_sector_overlap_right"] = right_ov
    out["trap_top_sector_overlap_lr_geom"] = combine_lr(left_ov, right_ov, "geom")
    out["trap_top_sector_overlap_lr_min"] = combine_lr(left_ov, right_ov, "min")
    out["trap_top_sector_overlap_lr_mean"] = combine_lr(left_ov, right_ov, "mean")
    out["trap_top_sector_overlap_lr_max"] = combine_lr(left_ov, right_ov, "max")
    out["trap_top_sector_overlap_lr_product"] = combine_lr(left_ov, right_ov, "product")
    return out


def _localization_key(family: str, vectors: str, side: str):
    fam = "pt" if family == "porter_thomas" else "uniform"
    if side in {"left", "right"}:
        return f"trap_q_{fam}_{side}_{vectors}"
    if side in {"mean", "geom", "min", "max", "product"}:
        return None
    raise ValueError(f"Unknown localization side: {side}")


def _get_localization_value(components, family="uniform", vectors="perm", side="right", domain="standard"):
    if domain == "fft":
        fam = "q_pt" if family == "porter_thomas" else "q_uniform"
        if side in {"left", "right"}:
            return safe_float(components.get(f"trap_fft_{fam}_{side}_{vectors}", np.nan))
        left = safe_float(components.get(f"trap_fft_{fam}_left_{vectors}", np.nan))
        right = safe_float(components.get(f"trap_fft_{fam}_right_{vectors}", np.nan))
        return combine_lr(left, right, side)

    key = _localization_key(family, vectors, side)
    if key is not None:
        return safe_float(components.get(key, np.nan))
    left = safe_float(components.get(f"trap_q_{'pt' if family == 'porter_thomas' else 'uniform'}_left_{vectors}", np.nan))
    right = safe_float(components.get(f"trap_q_{'pt' if family == 'porter_thomas' else 'uniform'}_right_{vectors}", np.nan))
    return combine_lr(left, right, side)


def _get_overlap_value(components, side="right", domain="standard", fft_overlap_measure="top_frequency_mass", vectors="perm"):
    if domain == "fft":
        base = "trap_fft_top_frequency_mass" if fft_overlap_measure == "top_frequency_mass" else "trap_fft_selected_frequency_mass"
        if side in {"left", "right"}:
            return safe_float(components.get(f"{base}_{side}_{vectors}", np.nan))
        left = safe_float(components.get(f"{base}_left_{vectors}", np.nan))
        right = safe_float(components.get(f"{base}_right_{vectors}", np.nan))
        return combine_lr(left, right, side)
    if side in {"left", "right"}:
        return safe_float(components.get(f"trap_top_sector_overlap_{side}", np.nan))
    if side in {"mean", "geom", "min", "max", "product"}:
        return safe_float(components.get(f"trap_top_sector_overlap_lr_{side}", np.nan))
    raise ValueError(f"Unknown overlap side: {side}")


def compute_burden_variant(components, config):
    spectral_mode = config.get("spectral_mode", "edge_ratio_current")
    spectral_power = float(config.get("spectral_power", 1.0))
    localization_family = config.get("localization_family", "uniform")
    localization_vectors = config.get("localization_vectors", "perm")
    localization_side = config.get("localization_side", "right")
    localization_power = float(config.get("localization_power", 1.0))
    overlap_side = config.get("overlap_side", "right")
    overlap_power = float(config.get("overlap_power", 1.0))
    localization_domain = config.get("localization_domain", "standard")
    overlap_domain = config.get("overlap_domain", "standard")
    fft_localization_family = config.get("fft_localization_family", "uniform")
    fft_overlap_measure = config.get("fft_overlap_measure", "top_frequency_mass")
    overlap_vectors = config.get("overlap_vectors", localization_vectors)

    spectral_key = {
        "edge_ratio_current": "trap_spectral_edge_ratio_current",
        "edge_ratio_linear": "trap_spectral_edge_ratio_current",
        "total_excess": "trap_spectral_total_excess",
        "total_fraction": "trap_spectral_total_fraction",
        "raw_excess": "trap_spectral_raw_excess",
        "log_edge_ratio": "trap_spectral_log_edge_ratio",
    }.get(spectral_mode)
    if spectral_key is None:
        raise ValueError(f"Unknown spectral mode: {spectral_mode}")
    spectral_value = safe_float(components.get(spectral_key, np.nan))
    localization_value = _get_localization_value(
        components,
        family=(fft_localization_family if localization_domain == "fft" else localization_family),
        vectors=localization_vectors,
        side=localization_side,
        domain=localization_domain,
    )
    overlap_value = _get_overlap_value(
        components,
        side=overlap_side,
        domain=overlap_domain,
        fft_overlap_measure=fft_overlap_measure,
        vectors=overlap_vectors,
    )

    values = [spectral_value, localization_value, overlap_value]
    if any(not np.isfinite(v) for v in values):
        return float(np.nan)
    return float((spectral_value ** spectral_power) * (localization_value ** localization_power) * (overlap_value ** overlap_power))


DEFAULT_BURDEN_VARIANTS: List[Dict] = [
    dict(
        name="current_pr358",
        spectral_mode="edge_ratio_current",
        spectral_power=2,
        localization_family="porter_thomas",
        localization_vectors="perm",
        localization_side="right",
        localization_power=1,
        overlap_side="right",
        overlap_power=2,
    ),
    dict(
        name="edge_linear_uniform_right",
        spectral_mode="edge_ratio_current",
        spectral_power=1,
        localization_family="uniform",
        localization_vectors="perm",
        localization_side="right",
        localization_power=1,
        overlap_side="right",
        overlap_power=2,
    ),
    dict(
        name="edge_squared_pt_right_perm",
        spectral_mode="edge_ratio_current",
        spectral_power=2,
        localization_family="porter_thomas",
        localization_vectors="perm",
        localization_side="right",
        localization_power=1,
        overlap_side="right",
        overlap_power=2,
    ),
    dict(
        name="total_excess_pt_right_perm",
        spectral_mode="total_excess",
        spectral_power=1,
        localization_family="porter_thomas",
        localization_vectors="perm",
        localization_side="right",
        localization_power=1,
        overlap_side="right",
        overlap_power=2,
    ),
    dict(
        name="total_excess_pt_lr_geom_perm_overlap_lr_geom",
        spectral_mode="total_excess",
        spectral_power=1,
        localization_family="porter_thomas",
        localization_vectors="perm",
        localization_side="geom",
        localization_power=1,
        overlap_side="geom",
        overlap_power=1,
    ),
    dict(
        name="total_excess_pt_lr_min_perm_overlap_lr_min",
        spectral_mode="total_excess",
        spectral_power=1,
        localization_family="porter_thomas",
        localization_vectors="perm",
        localization_side="min",
        localization_power=1,
        overlap_side="min",
        overlap_power=1,
    ),
    dict(
        name="total_excess_pt_lr_geom_trap_overlap_lr_geom",
        spectral_mode="total_excess",
        spectral_power=1,
        localization_family="porter_thomas",
        localization_vectors="trap",
        localization_side="geom",
        localization_power=1,
        overlap_side="geom",
        overlap_power=1,
    ),
    dict(
        name="total_fraction_pt_lr_geom_perm_overlap_lr_geom",
        spectral_mode="total_fraction",
        spectral_power=1,
        localization_family="porter_thomas",
        localization_vectors="perm",
        localization_side="geom",
        localization_power=1,
        overlap_side="geom",
        overlap_power=1,
    ),
    dict(
        name="log_edge_pt_lr_geom_perm_overlap_lr_geom",
        spectral_mode="log_edge_ratio",
        spectral_power=1,
        localization_family="porter_thomas",
        localization_vectors="perm",
        localization_side="geom",
        localization_power=1,
        overlap_side="geom",
        overlap_power=1,
    ),
    dict(
        name="total_excess_pt_lr_geom_no_overlap",
        spectral_mode="total_excess",
        spectral_power=1,
        localization_family="porter_thomas",
        localization_vectors="perm",
        localization_side="geom",
        localization_power=1,
        overlap_side="right",
        overlap_power=0,
    ),
]

FFT_DEFAULT_BURDEN_VARIANTS: List[Dict] = [
    dict(
        name="fft_uniform_right_current_spectral",
        spectral_mode="edge_ratio_current",
        spectral_power=2,
        localization_domain="fft",
        fft_localization_family="uniform",
        localization_vectors="perm",
        localization_side="right",
        overlap_domain="standard",
        overlap_side="right",
        overlap_power=2,
    ),
    dict(
        name="fft_uniform_lr_geom_current_spectral",
        spectral_mode="edge_ratio_current",
        spectral_power=2,
        localization_domain="fft",
        fft_localization_family="uniform",
        localization_vectors="perm",
        localization_side="geom",
        overlap_domain="standard",
        overlap_side="right",
        overlap_power=2,
    ),
    dict(
        name="fft_uniform_lr_geom_fft_topmass",
        spectral_mode="edge_ratio_current",
        spectral_power=2,
        localization_domain="fft",
        fft_localization_family="uniform",
        localization_vectors="perm",
        localization_side="geom",
        overlap_domain="fft",
        fft_overlap_measure="top_frequency_mass",
        overlap_vectors="perm",
        overlap_side="geom",
        overlap_power=1,
    ),
    dict(
        name="fft_pt_lr_geom_fft_topmass",
        spectral_mode="edge_ratio_current",
        spectral_power=2,
        localization_domain="fft",
        fft_localization_family="porter_thomas",
        localization_vectors="perm",
        localization_side="geom",
        overlap_domain="fft",
        fft_overlap_measure="top_frequency_mass",
        overlap_vectors="perm",
        overlap_side="geom",
        overlap_power=1,
    ),
    dict(
        name="fft_uniform_total_fraction_lr_geom_fft_topmass",
        spectral_mode="total_fraction",
        spectral_power=1,
        localization_domain="fft",
        fft_localization_family="uniform",
        localization_vectors="perm",
        localization_side="geom",
        overlap_domain="fft",
        fft_overlap_measure="top_frequency_mass",
        overlap_vectors="perm",
        overlap_side="geom",
        overlap_power=1,
    ),
    dict(
        name="fft_pt_total_fraction_lr_geom_fft_topmass",
        spectral_mode="total_fraction",
        spectral_power=1,
        localization_domain="fft",
        fft_localization_family="porter_thomas",
        localization_vectors="perm",
        localization_side="geom",
        overlap_domain="fft",
        fft_overlap_measure="top_frequency_mass",
        overlap_vectors="perm",
        overlap_side="geom",
        overlap_power=1,
    ),
]


def resolve_burden_variant_configs(burden_variants, trap_fft=False):
    if burden_variants is None:
        return None
    if burden_variants == "default":
        out = copy.deepcopy(DEFAULT_BURDEN_VARIANTS)
        if bool(trap_fft):
            out.extend(copy.deepcopy(FFT_DEFAULT_BURDEN_VARIANTS))
        return out
    if isinstance(burden_variants, list):
        return copy.deepcopy(burden_variants)
    raise ValueError("burden_variants must be None, 'default', or list[dict]")


def compute_burden_variants(components, variant_configs):
    out = {}
    if variant_configs is None:
        return out
    for cfg in variant_configs:
        name = cfg.get("name")
        if not name:
            raise ValueError("Each burden variant config must include a non-empty 'name'")
        out[f"trap_variance_burden__{name}"] = compute_burden_variant(components, cfg)
    return out


def _confusion_from_binary(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def evaluate_burden_variants(
    trap_df,
    label_col,
    variant_cols=None,
    positive_label=1,
    thresholds=None,
    top_k=None,
    group_col=None,
):
    df = trap_df.copy()
    if variant_cols is None:
        variant_cols = [c for c in df.columns if c.startswith("trap_variance_burden__")]
    if thresholds is None:
        thresholds = [0.0]
    if top_k is None:
        top_k = []

    y_true = (df[label_col] == positive_label).astype(int).to_numpy()
    rows = []
    for col in variant_cols:
        scores = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()

        auroc = np.nan
        auprc = np.nan
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
            if len(np.unique(y_true)) > 1:
                auroc = float(roc_auc_score(y_true, scores))
                auprc = float(average_precision_score(y_true, scores))
        except Exception:
            pass

        for th in thresholds:
            y_pred = (scores >= float(th)).astype(int)
            tp, fp, tn, fn = _confusion_from_binary(y_true, y_pred)
            n = int(len(y_true))
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else np.nan
            rows.append(
                dict(
                    variant=col,
                    threshold=float(th),
                    top_k=np.nan,
                    n=n,
                    tp=tp,
                    fp=fp,
                    tn=tn,
                    fn=fn,
                    false_positive_rate=fpr,
                    precision=precision,
                    recall=recall,
                    specificity=specificity,
                    false_positives=fp,
                    auroc=auroc,
                    auprc=auprc,
                )
            )

        ranked = np.argsort(-scores)
        for k in top_k:
            kk = int(min(max(int(k), 0), len(scores)))
            y_pred = np.zeros_like(y_true)
            if kk > 0:
                y_pred[ranked[:kk]] = 1
            tp, fp, tn, fn = _confusion_from_binary(y_true, y_pred)
            n = int(len(y_true))
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else np.nan
            rows.append(
                dict(
                    variant=col,
                    threshold=np.nan,
                    top_k=kk,
                    n=n,
                    tp=tp,
                    fp=fp,
                    tn=tn,
                    fn=fn,
                    false_positive_rate=fpr,
                    precision=precision,
                    recall=recall,
                    specificity=specificity,
                    false_positives=fp,
                    auroc=auroc,
                    auprc=auprc,
                )
            )

    return pd.DataFrame(rows)
