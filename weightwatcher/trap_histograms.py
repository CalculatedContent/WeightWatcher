import logging

import matplotlib.pyplot as plt
import numpy as np

from .RMT_Util import save_fig
from .constants import SAVEFIG, SAVEDIR, WW_NAME

logger = logging.getLogger(WW_NAME)


def _resolve_trap_assessment(trap_info):
    assessment = trap_info.get("trap_assessment", None)
    if assessment is not None:
        return assessment

    risk_score = float(trap_info.get("trap_risk_score", 0.0))
    if risk_score >= 0.5:
        return "localized_risky"
    if risk_score <= 0.2:
        return "benign_diffuse"
    return "mixed"


def _adjust_color_towards_white(color, amount):
    rgb = np.array(color[:3], dtype=float)
    adjusted = rgb + (1.0 - rgb) * float(np.clip(amount, 0.0, 1.0))
    return tuple(np.clip(adjusted, 0.0, 1.0))


def _adjust_color_towards_black(color, amount):
    rgb = np.array(color[:3], dtype=float)
    adjusted = rgb * (1.0 - float(np.clip(amount, 0.0, 1.0)))
    return tuple(np.clip(adjusted, 0.0, 1.0))


def _trap_color(trap_index, assessment):
    cmap = plt.get_cmap("tab20")
    base = cmap((trap_index - 1) % 20)

    if assessment == "localized_risky":
        return _adjust_color_towards_black(base, 0.25)
    if assessment == "benign_diffuse":
        return _adjust_color_towards_white(base, 0.35)
    return base[:3]


def _largest_trap_elements(trap_matrix, atol=1e-12):
    """Return entries of a trap component at its max-|value| locations.

    Note: a rank-1 trap component is only one additive term in the full matrix
    decomposition. Its entries can exceed the observed min/max of W because
    other components can cancel it when summed back into W.
    """
    abs_vals = np.abs(trap_matrix)
    if abs_vals.size == 0:
        return np.array([], dtype=float)

    max_abs = np.max(abs_vals)
    if max_abs <= 0.0:
        return np.array([], dtype=float)

    mask = np.isclose(abs_vals, max_abs, atol=atol, rtol=0.0)
    return trap_matrix[mask]


def _largest_trap_weight_values(trap_matrix, weight_matrix, atol=1e-12):
    """Map trap peak locations into the actual weight-matrix value space."""
    trap_matrix = np.asarray(trap_matrix, dtype=float)
    weight_matrix = np.asarray(weight_matrix, dtype=float)

    if trap_matrix.shape != weight_matrix.shape:
        return _largest_trap_elements(trap_matrix, atol=atol)

    abs_vals = np.abs(trap_matrix)
    if abs_vals.size == 0:
        return np.array([], dtype=float)

    max_abs = np.max(abs_vals)
    if max_abs <= 0.0:
        return np.array([], dtype=float)

    mask = np.isclose(abs_vals, max_abs, atol=atol, rtol=0.0)
    return weight_matrix[mask]


def _format_hist_value_label(value):
    return f"{float(value):.4g}"


def _significant_trap_components(trap_matrix, weight_matrix, min_rel_coeff=0.25, max_components=300):
    trap_flat = np.asarray(trap_matrix, dtype=float).ravel()
    weight_flat = np.asarray(weight_matrix, dtype=float).ravel()

    if trap_flat.size == 0 or weight_flat.size == 0 or trap_flat.size != weight_flat.size:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    abs_coeff = np.abs(trap_flat)
    max_abs = float(np.max(abs_coeff))
    if max_abs <= 0.0:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    rel_coeff = abs_coeff / max_abs
    mask = rel_coeff >= float(min_rel_coeff)
    if not np.any(mask):
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    x_vals = weight_flat[mask]
    rel_vals = rel_coeff[mask]
    signed_coeff = trap_flat[mask]

    if x_vals.size > max_components:
        order = np.argsort(rel_vals)[::-1][:max_components]
        x_vals = x_vals[order]
        rel_vals = rel_vals[order]
        signed_coeff = signed_coeff[order]

    return x_vals, rel_vals, signed_coeff


def plot_layer_trap_weight_histogram(ww_layer, trap_infos, params=None, method_tag="analyze_traps"):
    """Plot per-layer weight histograms with dashed vertical trap marker lines.

    Parameters
    ----------
    ww_layer : WWLayer
        Layer to plot.
    trap_infos : list[dict]
        Each dict should include `trap_index`, `trap_matrix` and optionally
        `trap_assessment` / `trap_risk_score`.
    params : dict
        WeightWatcher parameter dictionary.
    method_tag : str
        Method name included in titles and save-file labels.
    """

    if trap_infos is None or len(trap_infos) == 0:
        return

    if len(ww_layer.Wmats) == 0:
        return

    weights = ww_layer.Wmats[0]
    if weights is None:
        return

    flat_weights = np.asarray(weights, dtype=float).ravel()
    if flat_weights.size == 0:
        return

    layer_label = f"Layer {ww_layer.layer_id}"
    if getattr(ww_layer, "name", None):
        layer_label = f"{layer_label} ({ww_layer.name})"

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(flat_weights, bins=100, density=True, alpha=0.55, color="steelblue")
    y_max = float(ax.get_ylim()[1])
    trap_bar_threshold = 0.30

    min_w = float(np.min(flat_weights))
    max_w = float(np.max(flat_weights))
    indicator_height = max(y_max * 0.08, 1e-12)
    ax.vlines(
        [min_w, max_w],
        ymin=0.0,
        ymax=indicator_height,
        colors="black",
        linewidth=1.4,
        alpha=0.85,
    )

    drawn_labels = set()
    trap_label_count = 0
    for trap_info in trap_infos:
        trap_index = int(trap_info["trap_index"])
        assessment = _resolve_trap_assessment(trap_info)
        line_color = _trap_color(trap_index, assessment)

        x_vals, rel_vals, signed_coeff = _significant_trap_components(
            np.asarray(trap_info["trap_matrix"], dtype=float),
            np.asarray(weights, dtype=float),
            min_rel_coeff=trap_bar_threshold,
        )
        if x_vals.size == 0:
            continue

        label = None
        if trap_index not in drawn_labels:
            label = f"trap {trap_index}"
            drawn_labels.add(trap_index)

        ax.vlines(
            x_vals,
            ymin=0.0,
            ymax=rel_vals * y_max,
            colors=[line_color],
            linewidth=1.0,
            alpha=0.35,
            label=label,
        )

        top_idx = int(np.argmax(rel_vals))
        top_x = float(x_vals[top_idx])
        top_coeff = float(signed_coeff[top_idx])
        text_y = y_max * max(0.42, 0.98 - 0.07 * (trap_label_count % 7))
        ax.text(
            top_x,
            text_y,
            _format_hist_value_label(top_coeff),
            color=line_color,
            rotation=90,
            va="top",
            ha="right",
            fontsize=7,
            alpha=0.95,
        )
        trap_label_count += 1

    ax.set_xlabel("Weight value")
    ax.set_ylabel("Density")
    ax.set_title(f"{layer_label} — {method_tag} trap lines on weight histogram")
    if len(drawn_labels) > 0:
        ax.legend(loc="best", fontsize=8)

    if params is None:
        params = {}

    if params.get(SAVEFIG, False):
        savedir = params.get(SAVEDIR, "ww-img")
        save_fig(plt, f"{method_tag}.trap_hist", ww_layer.plot_id, savedir)
    else:
        plt.tight_layout()

    plt.close(fig)
