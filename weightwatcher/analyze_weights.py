import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from .constants import *

LAPLACE = "laplace"
MIN_LOGNORMAL_SIDE_SAMPLES = 64
LOGNORMAL_KS_PVALUE_THRESHOLD = 0.05
LOGNORMAL_NSA_TOL = 0.25


def _side_column_name(side_name):
    return "right" if side_name == "right" else "left"


def _default_lognormal_side_stats(side_name):
    side = _side_column_name(side_name)
    return {
        f"lognormal_{side}_detected": False,
        f"lognormal_{side}_mu": np.nan,
        f"lognormal_{side}_sigma2": np.nan,
        f"lognormal_{side}_n": np.nan,
        f"lognormal_{side}_log_n": np.nan,
        f"lognormal_{side}_sa_ratio": np.nan,
        f"lognormal_{side}_nsa_margin": np.nan,
        f"lognormal_{side}_sa_regime": "undetermined",
        f"lognormal_{side}_non_self_averaging": np.nan,
    }


def compute_lognormal_self_averaging_stats(values, side_name, min_samples, tol, classified_as_lognormal=True):
    side = _side_column_name(side_name)
    stats_row = _default_lognormal_side_stats(side_name)

    vals = np.asarray(values).ravel().astype(float)
    vals = vals[np.isfinite(vals)]
    if side == "right":
        x = vals[vals > 0]
    else:
        x = np.abs(vals[vals < 0])
        x = x[x > 0]

    if len(x) < min_samples or not classified_as_lognormal:
        return stats_row

    try:
        ln_params = stats.lognorm.fit(x, floc=0)
        _, ks_p, _ = _ks_metrics(x, LOG_NORMAL, ln_params)
    except Exception:
        return stats_row

    plausible = np.isfinite(ks_p) and ks_p >= LOGNORMAL_KS_PVALUE_THRESHOLD
    if not plausible:
        return stats_row

    log_x = np.log(x)
    mu_hat = float(np.mean(log_x))
    sigma2_hat = float(np.var(log_x))
    n_side = int(len(x))
    log_n = float(np.log(n_side))
    sa_ratio = float(np.exp(sigma2_hat) / n_side)
    nsa_margin = float(sigma2_hat - log_n)

    if sigma2_hat < log_n - tol:
        regime = "self_averaging"
        nsa_bool = False
    elif sigma2_hat > log_n + tol:
        regime = "non_self_averaging"
        nsa_bool = True
    else:
        regime = "marginal"
        nsa_bool = False

    stats_row.update({
        f"lognormal_{side}_detected": True,
        f"lognormal_{side}_mu": mu_hat,
        f"lognormal_{side}_sigma2": sigma2_hat,
        f"lognormal_{side}_n": n_side,
        f"lognormal_{side}_log_n": log_n,
        f"lognormal_{side}_sa_ratio": sa_ratio,
        f"lognormal_{side}_nsa_margin": nsa_margin,
        f"lognormal_{side}_sa_regime": regime,
        f"lognormal_{side}_non_self_averaging": nsa_bool,
    })

    return stats_row


def _ks_metrics(data, dist_name, params):
    if len(data) < 8:
        return np.nan, np.nan, np.nan

    if dist_name == POWER_LAW:
        cdf = lambda x: stats.pareto.cdf(x, *params)
        logpdf = stats.pareto.logpdf(data, *params)
    elif dist_name == EXPONENTIAL:
        cdf = lambda x: stats.expon.cdf(x, *params)
        logpdf = stats.expon.logpdf(data, *params)
    elif dist_name == LOG_NORMAL:
        cdf = lambda x: stats.lognorm.cdf(x, *params)
        logpdf = stats.lognorm.logpdf(data, *params)
    elif dist_name == LAPLACE:
        cdf = lambda x: stats.laplace.cdf(x, *params)
        logpdf = stats.laplace.logpdf(data, *params)
    else:
        return np.nan, np.nan, np.nan

    ks_stat, ks_p = stats.kstest(data, cdf)
    ll = float(np.sum(logpdf[np.isfinite(logpdf)]))
    return float(ks_stat), float(ks_p), ll


def _fit_side_models(side_values, side_label, min_points=32):
    vals = np.asarray(side_values).ravel().astype(float)
    vals = vals[np.isfinite(vals)]

    if side_label == "left":
        signed = vals[vals < 0]
        mags = np.abs(signed)
    else:
        signed = vals[vals > 0]
        mags = signed.copy()

    mags = mags[mags > 0]
    signed = signed[np.abs(signed) > 0]

    rows = []
    if len(mags) < min_points:
        return rows

    try:
        pl_params = stats.pareto.fit(mags, floc=0)
        ks, p, ll = _ks_metrics(mags, POWER_LAW, pl_params)
        rows.append({"side": side_label, "distribution": POWER_LAW, "params": {"shape": float(pl_params[0]), "scale": float(pl_params[2])}, "ks_stat": ks, "ks_pvalue": p, "log_likelihood": ll, "n": int(len(mags))})
    except Exception:
        pass

    try:
        exp_params = stats.expon.fit(mags, floc=0)
        ks, p, ll = _ks_metrics(mags, EXPONENTIAL, exp_params)
        rows.append({"side": side_label, "distribution": EXPONENTIAL, "params": {"scale": float(exp_params[1])}, "ks_stat": ks, "ks_pvalue": p, "log_likelihood": ll, "n": int(len(mags))})
    except Exception:
        pass

    try:
        ln_params = stats.lognorm.fit(mags, floc=0)
        ks, p, ll = _ks_metrics(mags, LOG_NORMAL, ln_params)
        rows.append({"side": side_label, "distribution": LOG_NORMAL, "params": {"sigma": float(ln_params[0]), "scale": float(ln_params[2])}, "ks_stat": ks, "ks_pvalue": p, "log_likelihood": ll, "n": int(len(mags))})
    except Exception:
        pass

    try:
        # Laplace on signed side values.
        if len(signed) >= min_points:
            loc = float(np.median(signed))
            scale = float(np.mean(np.abs(signed - loc)))
            lap_params = (loc, max(scale, 1e-12))
            ks, p, ll = _ks_metrics(signed, LAPLACE, lap_params)
            rows.append({"side": side_label, "distribution": LAPLACE, "params": {"loc": lap_params[0], "scale": lap_params[1]}, "ks_stat": ks, "ks_pvalue": p, "log_likelihood": ll, "n": int(len(signed))})
    except Exception:
        pass

    if len(rows) == 0:
        return rows

    sortable = [r for r in rows if np.isfinite(r["ks_stat"]) and np.isfinite(r["log_likelihood"])]
    if len(sortable) == 0:
        return rows

    best = sorted(sortable, key=lambda r: (r["ks_stat"], -r["log_likelihood"]))[0]["distribution"]
    for r in rows:
        r["is_best_fit"] = r["distribution"] == best
        r["best_fit_for_side"] = best

    return rows


def _plot_side_distribution(values, side_label, layer_id, layer_name, savefig=None):
    vals = np.asarray(values).ravel().astype(float)
    vals = vals[np.isfinite(vals)]

    if side_label == "left":
        mags = np.abs(vals[vals < 0])
    else:
        mags = vals[vals > 0]

    mags = mags[mags > 0]
    if len(mags) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(mags, bins=80, alpha=0.8)
    axes[0].set_title(f"{side_label} distribution | layer {layer_id}: {layer_name}")
    axes[0].set_xlabel("magnitude")

    axes[1].hist(mags, bins=80, alpha=0.8, log=True)
    axes[1].set_xscale("log")
    axes[1].set_title(f"{side_label} log-log histogram")
    axes[1].set_xlabel("magnitude (log)")

    fig.tight_layout()

    if savefig:
        os.makedirs(savefig, exist_ok=True)
        path = os.path.join(savefig, f"weights_dist_layer{layer_id}_{side_label}.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze_weights(
    watcher,
    model=None,
    layers=[],
    min_evals=DEFAULT_MIN_EVALS,
    max_evals=DEFAULT_MAX_EVALS,
    min_size=None,
    max_size=None,
    max_N=DEFAULT_MAX_N,
    glorot_fix=False,
    plot=False,
    savefig=DEF_SAVE_DIR,
    conv2d_norm=True,
    ww2x=DEFAULT_WW2X,
    pool=DEFAULT_POOL,
    conv2d_fft=False,
    fft=False,
    channels=None,
    start_ids=DEFAULT_START_ID,
    base_model=None,
    peft=DEFAULT_PEFT,
    fast=False,
    sample_size=100000,
    random_state=123,
):
    """Analyze weight-entry distributions by layer.

    Fits left and right sides of each layer's weight distribution to:
    power law, exponential, log-normal, and Laplace.
    """

    watcher.set_model_(model, base_model)

    if min_size or max_size:
        pass

    params = DEFAULT_PARAMS.copy()
    params[MIN_EVALS] = min_evals
    params[MAX_EVALS] = max_evals
    params[MAX_N] = max_N
    params[GLOROT_FIT] = glorot_fix
    params[CONV2D_NORM] = conv2d_norm
    params[POOL] = pool
    params[WW2X] = ww2x
    params[CONV2D_FFT] = conv2d_fft
    params[FFT] = fft
    params[CHANNELS_STR] = channels
    params[LAYERS] = layers
    params[START_IDS] = start_ids
    params[PEFT] = peft

    params = watcher.normalize_params(params)

    layer_iterator = watcher.make_layer_iterator(model=watcher.model, layers=layers, params=params, base_model=watcher.base_model)

    rows = []
    rng = np.random.default_rng(random_state)

    for ww_layer in layer_iterator:
        if ww_layer.skipped or not ww_layer.has_weights:
            continue

        watcher.apply_normalize_Wmats(ww_layer, params)
        if params[FFT]:
            watcher.apply_FFT(ww_layer, params)

        if ww_layer.Wmats is None or len(ww_layer.Wmats) == 0:
            continue

        layer_values = np.concatenate([np.asarray(w).ravel() for w in ww_layer.Wmats]).astype(float)
        layer_values = layer_values[np.isfinite(layer_values)]

        if fast and len(layer_values) > sample_size:
            ids = rng.choice(len(layer_values), size=sample_size, replace=False)
            layer_values = layer_values[ids]

        layer_rows = []
        side_best = {}
        for side in ("left", "right"):
            side_rows = _fit_side_models(layer_values, side_label=side)
            side_best[side] = np.nan
            for row in side_rows:
                row["layer_id"] = ww_layer.layer_id
                row["name"] = ww_layer.name
                row["longname"] = ww_layer.longname
                row["layer_type"] = str(ww_layer.the_type)
                layer_rows.append(row)
                if row.get("is_best_fit", False):
                    side_best[side] = row.get("distribution")

            if plot:
                _plot_side_distribution(layer_values, side, ww_layer.layer_id, ww_layer.name, savefig=savefig)

        right_sa = compute_lognormal_self_averaging_stats(
            layer_values,
            side_name="right",
            min_samples=MIN_LOGNORMAL_SIDE_SAMPLES,
            tol=LOGNORMAL_NSA_TOL,
            classified_as_lognormal=(side_best.get("right") == LOG_NORMAL),
        )
        left_sa = compute_lognormal_self_averaging_stats(
            layer_values,
            side_name="left",
            min_samples=MIN_LOGNORMAL_SIDE_SAMPLES,
            tol=LOGNORMAL_NSA_TOL,
            classified_as_lognormal=(side_best.get("left") == LOG_NORMAL),
        )

        layer_diag = {}
        layer_diag.update(right_sa)
        layer_diag.update(left_sa)
        for row in layer_rows:
            row.update(layer_diag)
            rows.append(row)

    details = pd.DataFrame(rows)
    if len(details) > 0:
        lead_cols = ["layer_id", "name", "side", "distribution", "is_best_fit", "best_fit_for_side"]
        lead_cols = [c for c in lead_cols if c in details.columns]
        details = details[lead_cols + [c for c in details.columns if c not in lead_cols]]

    watcher.details = details
    return details
