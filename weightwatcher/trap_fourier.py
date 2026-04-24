import numpy as np

DEFAULT_TRAP_FFT_CONFIG = {
    "sides": "both",
    "vectors": "both",
    "fold_conjugates": True,
    "exclude_dc": False,
    "top_frequency_l": 1,
    "selected_frequencies": None,
    "normalization": "ortho",
    "baseline": "uniform",  # uniform | pt_real_mc | pt_complex
    "mc_samples": 2048,
    "mc_seed": 123,
    "modulus": None,
    "apply_only_if_length_matches_modulus": False,
    "layer_fft_map": None,
}

_PT_CACHE = {}


def resolve_trap_fft_config(cfg=None):
    out = dict(DEFAULT_TRAP_FFT_CONFIG)
    if cfg:
        out.update(dict(cfg))
    out["top_frequency_l"] = int(out.get("top_frequency_l", 1))
    if out["top_frequency_l"] < 1:
        raise ValueError("trap_fft_config['top_frequency_l'] must be >= 1")
    return out


def normalize_vector(vec):
    v = np.asarray(vec).ravel()
    if v.size == 0:
        return None
    if not np.iscomplexobj(v):
        v = v.astype(float)
    norm = np.linalg.norm(v)
    if (not np.isfinite(norm)) or norm <= 0.0:
        return None
    v = v / norm
    if not np.all(np.isfinite(np.real(v))) or not np.all(np.isfinite(np.imag(v))):
        return None
    return v


def _fold_conjugates(mass):
    n = len(mass)
    if n == 0:
        return np.array([], dtype=float)
    out = [float(mass[0])]
    for k in range(1, (n - 1) // 2 + 1):
        out.append(float(mass[k] + mass[n - k]))
    if n % 2 == 0:
        out.append(float(mass[n // 2]))
    out = np.asarray(out, dtype=float)
    s = np.sum(out)
    if np.isfinite(s) and s > 0:
        out = out / s
    return out


def fourier_mass(vec, fold_conjugates=True, exclude_dc=False, normalization="ortho"):
    v = normalize_vector(vec)
    if v is None:
        return {
            "fft_values": None,
            "mass": np.array([], dtype=float),
            "folded_mass": None,
            "effective_mass": np.array([], dtype=float),
            "n": 0,
            "n_effective": 0,
        }

    fv = np.fft.fft(v, norm=normalization)
    mass = np.abs(fv) ** 2
    s = np.sum(mass)
    if not np.isfinite(s) or s <= 0:
        return {
            "fft_values": fv,
            "mass": np.array([], dtype=float),
            "folded_mass": None,
            "effective_mass": np.array([], dtype=float),
            "n": len(v),
            "n_effective": 0,
        }
    mass = mass / s
    folded = _fold_conjugates(mass) if fold_conjugates else None
    effective = folded if fold_conjugates else mass.copy()
    if exclude_dc and len(effective) > 1:
        effective = effective.copy()
        effective[0] = 0.0
        e = np.sum(effective)
        if np.isfinite(e) and e > 0:
            effective = effective / e

    return {
        "fft_values": fv,
        "mass": np.asarray(mass, dtype=float),
        "folded_mass": None if folded is None else np.asarray(folded, dtype=float),
        "effective_mass": np.asarray(effective, dtype=float),
        "n": int(len(v)),
        "n_effective": int(len(effective)),
    }


def fourier_ipr(vec, **kwargs):
    fm = fourier_mass(vec, **kwargs)
    p = fm["effective_mass"]
    if p.size == 0:
        return float(np.nan)
    return float(np.sum(p ** 2))


def fourier_uniform_centered_q(vec, **kwargs):
    fm = fourier_mass(vec, **kwargs)
    p = fm["effective_mass"]
    n_eff = fm["n_effective"]
    if p.size == 0:
        return float(np.nan), float(np.nan)
    ipr = float(np.sum(p ** 2))
    if n_eff <= 1:
        return ipr, 1.0
    q = (n_eff * ipr - 1.0) / (n_eff - 1.0)
    return ipr, float(np.clip(q, 0.0, 1.0))


def _expected_ipr_real_mc(n, fold_conjugates, exclude_dc, normalization, mc_samples, mc_seed):
    key = (int(n), bool(fold_conjugates), bool(exclude_dc), str(normalization), int(mc_samples), int(mc_seed))
    if key in _PT_CACHE:
        return _PT_CACHE[key]
    rng = np.random.RandomState(int(mc_seed))
    vals = []
    for _ in range(int(mc_samples)):
        v = rng.normal(size=int(n))
        vals.append(fourier_ipr(v, fold_conjugates=fold_conjugates, exclude_dc=exclude_dc, normalization=normalization))
    expected = float(np.nanmean(vals)) if len(vals) else float(np.nan)
    _PT_CACHE[key] = expected
    return expected


def fourier_pt_centered_q(vec, baseline="pt_real_mc", mc_samples=2048, mc_seed=123, **kwargs):
    fm = fourier_mass(vec, **kwargs)
    p = fm["effective_mass"]
    n_eff = fm["n_effective"]
    if p.size == 0:
        return float(np.nan), float(np.nan)
    ipr = float(np.sum(p ** 2))
    if n_eff <= 1:
        return ipr, 1.0

    if baseline == "uniform":
        expected = 1.0 / n_eff
    elif baseline == "pt_complex":
        expected = 2.0 / (n_eff + 1.0)
    elif baseline == "pt_real_mc":
        expected = _expected_ipr_real_mc(
            fm["n"],
            kwargs.get("fold_conjugates", True),
            kwargs.get("exclude_dc", False),
            kwargs.get("normalization", "ortho"),
            mc_samples,
            mc_seed,
        )
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    if (not np.isfinite(expected)) or expected >= 1.0:
        return ipr, float(np.nan)
    q = (ipr - expected) / (1.0 - expected)
    return ipr, float(np.clip(q, 0.0, 1.0))


def fourier_top_frequency_mass(vec, top_frequency_l=1, **kwargs):
    fm = fourier_mass(vec, **kwargs)
    p = fm["effective_mass"]
    if p.size == 0:
        return float(np.nan), [], 0
    l = int(top_frequency_l)
    if l < 1:
        raise ValueError("top_frequency_l must be >= 1")
    l_eff = min(l, len(p))
    order = np.argsort(p)[::-1]
    idx = order[:l_eff]
    return float(np.sum(p[idx])), [int(i) for i in idx], int(l_eff)


def fourier_selected_frequency_mass(vec, selected_frequencies=None, **kwargs):
    if selected_frequencies is None:
        return float(np.nan)
    fm = fourier_mass(vec, **kwargs)
    p = fm["effective_mass"]
    if p.size == 0:
        return float(np.nan)
    idx = [int(i) for i in selected_frequencies if 0 <= int(i) < len(p)]
    return float(np.sum(p[idx])) if len(idx) else 0.0


def fourier_component_summary(vec, prefix, trap_fft_config):
    cfg = resolve_trap_fft_config(trap_fft_config)
    shared = dict(
        fold_conjugates=bool(cfg.get("fold_conjugates", True)),
        exclude_dc=bool(cfg.get("exclude_dc", False)),
        normalization=cfg.get("normalization", "ortho"),
    )
    ipr, q_uniform = fourier_uniform_centered_q(vec, **shared)
    _, q_pt = fourier_pt_centered_q(
        vec,
        baseline=cfg.get("baseline", "uniform"),
        mc_samples=int(cfg.get("mc_samples", 2048)),
        mc_seed=int(cfg.get("mc_seed", 123)),
        **shared,
    )
    top_mass, top_idx, l_eff = fourier_top_frequency_mass(
        vec,
        top_frequency_l=int(cfg.get("top_frequency_l", 1)),
        **shared,
    )
    selected_mass = fourier_selected_frequency_mass(
        vec,
        selected_frequencies=cfg.get("selected_frequencies", None),
        **shared,
    )

    peak_freq = top_idx[0] if len(top_idx) else np.nan
    peak_mass = top_mass if l_eff == 1 else np.nan

    return {
        f"{prefix}_fft_ipr": ipr,
        f"{prefix}_fft_q_uniform": q_uniform,
        f"{prefix}_fft_q_pt": q_pt,
        f"{prefix}_fft_top_frequency_mass": top_mass,
        f"{prefix}_fft_peak_frequency": peak_freq,
        f"{prefix}_fft_peak_mass": peak_mass,
        f"{prefix}_fft_selected_frequency_mass": selected_mass,
    }


def length_matches_modulus(vec_len, trap_fft_config):
    cfg = resolve_trap_fft_config(trap_fft_config)
    if not bool(cfg.get("apply_only_if_length_matches_modulus", False)):
        return True
    modulus = cfg.get("modulus", None)
    if modulus is None:
        return False
    m = int(modulus)
    allowed = {m, m + 1}
    return int(vec_len) in allowed
