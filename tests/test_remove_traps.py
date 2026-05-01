import numpy as np
import pandas as pd
import pytest

from weightwatcher.constants import CHANNELS, FRAMEWORK, LAYER_TYPE, DEFAULT_PARAMS
from weightwatcher.RMT_Util import svd_full
from weightwatcher.weightwatcher import FrameworkLayer, WWLayer, WeightWatcher
from weightwatcher.weightwatcher import PyTorchLayer

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class FakeDenseFrameworkLayer(FrameworkLayer):
    def __init__(self, W):
        self._W = W.copy()
        super().__init__(
            layer=self,
            layer_id=0,
            name="fake_dense",
            longname="fake_dense",
            the_type=LAYER_TYPE.DENSE,
            framework=FRAMEWORK.PYTORCH,
            channels=CHANNELS.FIRST,
            has_bias=False,
        )

    def get_weights_and_biases(self):
        return True, self._W.copy(), False, None

    def replace_layer_weights(self, W, B=None):
        self._W = W.copy()


def random_unit_vector(rng, n):
    x = rng.standard_normal(n)
    return x / np.linalg.norm(x)


def orthogonalize_and_normalize(vec, basis_vec):
    out = vec - np.dot(vec, basis_vec) * basis_vec
    return out / np.linalg.norm(out)


def planted_rank1(u, v, strength):
    return strength * np.outer(u, v)


def abs_overlap(x, y):
    return np.abs(np.dot(x / np.linalg.norm(x), y / np.linalg.norm(y)))


def top_singular_vectors(X):
    u, _, vh = svd_full(X)
    return u[:, 0], vh[0, :]


def make_test_params():
    params = DEFAULT_PARAMS.copy()
    params["pool"] = True
    params["plot"] = False
    params["normalize"] = False
    return params


def make_ww_layer(W):
    return WWLayer(FakeDenseFrameworkLayer(W), params=make_test_params())


def _single_trap_setup(seed=1234, n=96, noise_std=0.04, s1=10.0):
    rng = np.random.default_rng(seed)
    W_base = rng.normal(0.0, noise_std, size=(n, n))
    u1 = np.zeros(n)
    v1 = np.zeros(n)
    u1[7] = 1.0
    v1[61] = 1.0
    T1 = planted_rank1(u1, v1, s1)
    W = W_base + T1
    return W, T1, u1, v1


def _two_trap_setup(seed=7, n=96, noise_std=0.03, s1=12.0, s2=8.5):
    rng = np.random.default_rng(seed)
    W_base = rng.normal(0.0, noise_std, size=(n, n))

    u1 = np.zeros(n)
    v1 = np.zeros(n)
    u2 = np.zeros(n)
    v2 = np.zeros(n)
    u1[5] = 1.0
    v1[20] = 1.0
    u2[44] = 1.0
    v2[80] = 1.0

    T1 = planted_rank1(u1, v1, s1)
    T2 = planted_rank1(u2, v2, s2)
    W = W_base + T1 + T2
    return W, (T1, T2), (u1, v1, u2, v2)


def test_remove_traps_single_trap_flow():
    watcher = WeightWatcher(model=None)
    W, T1, u1, v1 = _single_trap_setup()
    ww_layer = make_ww_layer(W)

    artifacts = watcher._collect_trap_artifacts(ww_layer, params=make_test_params(), seed=101)
    assert len(artifacts) == 1

    art = artifacts[0]
    T_detect = art["T_orig_raw"]

    assert np.isclose(np.linalg.norm(T_detect, "fro"), np.linalg.norm(T1, "fro"), rtol=0.25)
    assert np.isclose(np.var(T_detect), np.var(T1), rtol=0.35)

    assert abs_overlap(art["u_trap"], u1) > 0.80
    assert abs_overlap(art["v_trap"], v1) > 0.80

    rng = np.random.default_rng(303)
    R = watcher._make_stat_matched_random_matrix(T_detect, rng)
    assert R.shape == T_detect.shape
    assert np.isclose(np.linalg.norm(R, "fro"), np.linalg.norm(T_detect, "fro"), rtol=1e-6, atol=1e-8)
    assert np.isclose(np.var(R), np.var(T_detect), rtol=1e-6, atol=1e-8)

    ru, rv = top_singular_vectors(R)
    assert abs_overlap(ru, u1) < abs_overlap(art["u_trap"], u1)
    assert abs_overlap(rv, v1) < abs_overlap(art["v_trap"], v1)

    ww_layer_remove = make_ww_layer(W)
    watcher.apply_remove_traps(ww_layer_remove, trap_indices=[1], params=make_test_params(), seed=909)
    W_new = ww_layer_remove.framework_layer._W

    post_artifacts = watcher._collect_trap_artifacts(make_ww_layer(W_new), params=make_test_params(), seed=101)
    assert len(post_artifacts) == 0

    assert np.isclose(np.linalg.norm(W_new, "fro"), np.linalg.norm(W, "fro"), rtol=0.08)

    ww_a = make_ww_layer(W)
    ww_b = make_ww_layer(W)
    watcher.apply_remove_traps(ww_a, trap_indices=[1], params=make_test_params(), seed=999)
    watcher.apply_remove_traps(ww_b, trap_indices=[1], params=make_test_params(), seed=999)
    assert np.allclose(ww_a.framework_layer._W, ww_b.framework_layer._W)

    ww_c = make_ww_layer(W)
    watcher.apply_remove_traps(ww_c, trap_indices=[1], params=make_test_params(), seed=1000)
    assert not np.allclose(ww_a.framework_layer._W, ww_c.framework_layer._W)


def test_remove_traps_two_trap_index_selective_behavior():
    watcher = WeightWatcher(model=None)
    W, (T1, T2), (u1, v1, u2, v2) = _two_trap_setup()

    artifacts = watcher._collect_trap_artifacts(make_ww_layer(W), params=make_test_params(), seed=88)
    assert len(artifacts) == 2

    a1, a2 = artifacts[0], artifacts[1]
    assert abs_overlap(a1["u_trap"], u1) > 0.75
    assert abs_overlap(a1["v_trap"], v1) > 0.75
    assert abs_overlap(a2["u_trap"], u2) > 0.70
    assert abs_overlap(a2["v_trap"], v2) > 0.70

    ww_1 = make_ww_layer(W)
    watcher.apply_remove_traps(ww_1, trap_indices=[1], params=make_test_params(), seed=10)
    art_1 = watcher._collect_trap_artifacts(make_ww_layer(ww_1.framework_layer._W), params=make_test_params(), seed=88)
    assert len(art_1) == 1
    assert abs_overlap(art_1[0]["u_trap"], u2) > 0.65
    assert abs_overlap(art_1[0]["u_trap"], u1) < 0.60

    ww_2 = make_ww_layer(W)
    watcher.apply_remove_traps(ww_2, trap_indices=[2], params=make_test_params(), seed=11)
    art_2 = watcher._collect_trap_artifacts(make_ww_layer(ww_2.framework_layer._W), params=make_test_params(), seed=88)
    assert len(art_2) == 1
    assert abs_overlap(art_2[0]["u_trap"], u1) > 0.65
    assert abs_overlap(art_2[0]["u_trap"], u2) < 0.60

    ww_both = make_ww_layer(W)
    watcher.apply_remove_traps(ww_both, trap_indices=[1, 2], params=make_test_params(), seed=12)
    art_both = watcher._collect_trap_artifacts(make_ww_layer(ww_both.framework_layer._W), params=make_test_params(), seed=88)
    assert len(art_both) == 0

    ww_bad = make_ww_layer(W)
    watcher.apply_remove_traps(ww_bad, trap_indices=[1, 2, 3], params=make_test_params(), seed=12)
    art_bad = watcher._collect_trap_artifacts(make_ww_layer(ww_bad.framework_layer._W), params=make_test_params(), seed=88)
    assert len(art_bad) == 0

    assert np.isclose(np.linalg.norm(ww_1.framework_layer._W, "fro"), np.linalg.norm(W, "fro"), rtol=0.08)
    assert np.isclose(np.linalg.norm(ww_2.framework_layer._W, "fro"), np.linalg.norm(W, "fro"), rtol=0.08)
    assert np.isclose(np.linalg.norm(ww_both.framework_layer._W, "fro"), np.linalg.norm(W, "fro"), rtol=0.10)


def test_remove_traps_public_api_direct_call(monkeypatch):
    W, _, _, _ = _single_trap_setup(seed=202)
    ww_layer = make_ww_layer(W)
    watcher = WeightWatcher(model={"dummy_weight": np.array([1.0])})

    monkeypatch.setattr(
        watcher,
        "make_layer_iterator",
        lambda model=None, layers=None, params=None, base_model=None: [ww_layer],
    )

    out_model = watcher.remove_traps(
        randomized_model={"dummy_weight": np.array([1.0])},
        layers=[],
        trap_indices=[1],
        seed=77,
        pool=True,
        plot=False,
    )
    assert isinstance(out_model, dict)

    W_new = ww_layer.framework_layer._W
    post_artifacts = watcher._collect_trap_artifacts(make_ww_layer(W_new), params=make_test_params(), seed=101)
    assert len(post_artifacts) == 0


def test_remove_traps_accepts_traps_dataframe_and_returns_verify_df(monkeypatch):
    W, _, _, _ = _single_trap_setup(seed=505)
    ww_layer = make_ww_layer(W)
    watcher = WeightWatcher(model={"dummy_weight": np.array([1.0])})

    monkeypatch.setattr(
        watcher,
        "make_layer_iterator",
        lambda model=None, layers=None, params=None, base_model=None: [ww_layer],
    )

    trap_df = pd.DataFrame([{
        "layer_id": 0, "trap_index": 1,
        "trap_variance_burden_old": 0.1,
        "trap_variance_burden_ipr": 0.2,
        "trap_variance_burden_top5": 0.3,
        "trap_variance_burden": 0.4,
    }])
    out_model, verify_df = watcher.remove_traps(
        randomized_model={"dummy_weight": np.array([1.0])},
        layers=[],
        traps=trap_df,
        seed=88,
        pool=True,
        plot=False,
        verify_traps=True,
        return_analyze=True,
    )
    assert isinstance(out_model, dict)
    assert isinstance(verify_df, pd.DataFrame)
    assert "verify_passed" in verify_df.columns
    assert "identity_verified" in verify_df.columns


def test_remove_traps_rejects_identity_mismatch(monkeypatch):
    W, _, _, _ = _single_trap_setup(seed=506)
    ww_layer = make_ww_layer(W)
    watcher = WeightWatcher(model={"dummy_weight": np.array([1.0])})
    monkeypatch.setattr(
        watcher,
        "make_layer_iterator",
        lambda model=None, layers=None, params=None, base_model=None: [ww_layer],
    )

    with pytest.raises(ValueError):
        watcher.remove_traps(
        randomized_model={"dummy_weight": np.array([1.0])},
            layers=[],
            traps=pd.DataFrame([{"layer_id": int(ww_layer.layer_id), "trap_index": 1, "trap_mode_index": 999999}]),
            seed=88,
            pool=True,
            plot=False,
        )


def test_remove_traps_invalid_indices_warns_and_skips(monkeypatch, caplog):
    W, _, _, _ = _single_trap_setup(seed=404)
    ww_layer = make_ww_layer(W)
    original_shape = ww_layer.framework_layer._W.shape
    watcher = WeightWatcher(model={"dummy_weight": np.array([1.0])})

    monkeypatch.setattr(
        watcher,
        "make_layer_iterator",
        lambda model=None, layers=None, params=None, base_model=None: [ww_layer],
    )

    caplog.set_level("WARNING")
    watcher.remove_traps(
        randomized_model={"dummy_weight": np.array([1.0])},
        layers=[],
        trap_indices=[1, 2, 3],
        seed=77,
        pool=True,
        plot=False,
    )

    assert any("Skipping invalid trap indices" in rec.message for rec in caplog.records)
    W_new = ww_layer.framework_layer._W
    assert W_new.shape == original_shape

    post_artifacts = watcher._collect_trap_artifacts(make_ww_layer(W_new), params=make_test_params(), seed=101)
    assert len(post_artifacts) == 0


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_trap_rng_consistency_analyze_vs_collect_single_and_multi_layer():
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 12, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(12, 10, bias=False),
    )
    with torch.no_grad():
        u1 = torch.linspace(1.0, 2.0, steps=12)
        v1 = torch.linspace(-2.0, 1.0, steps=16)
        model[0].weight.copy_(35.0 * torch.outer(u1, v1))

        u2 = torch.linspace(1.0, 1.5, steps=10)
        v2 = torch.linspace(-1.0, 2.0, steps=12)
        model[2].weight.copy_(20.0 * torch.outer(u2, v2))

    watcher = WeightWatcher(model=model)
    details = watcher.describe(model=model, pool=True)
    dense_layer_ids = details[details["layer_type"].astype(str).str.contains("dense", case=False)]["layer_id"].astype(int).tolist()
    assert len(dense_layer_ids) >= 2

    for layer_id in dense_layer_ids:
        trap_df = watcher.analyze_traps(model=model, layers=[layer_id], rng=123, plot=False, savefig=False, pool=True)
        expected_mode_indices = trap_df["perm_mode_index"].astype(int).tolist()

        params = DEFAULT_PARAMS.copy()
        params["pool"] = True
        params["plot"] = False
        params["layers"] = [layer_id]
        params = watcher.normalize_params(params)
        ww_layer = list(watcher.make_layer_iterator(model=watcher.model, layers=[layer_id], params=params))[0]
        artifacts = watcher._collect_trap_artifacts(ww_layer, params=params, seed=123)
        artifact_mode_indices = [int(a["trap_mode_index"]) for a in artifacts]

        assert len(expected_mode_indices) == len(artifact_mode_indices)
        assert expected_mode_indices == artifact_mode_indices


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_analyze_traps_rejects_default_rng_generator():
    model = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False))
    watcher = WeightWatcher(model=model)
    details = watcher.describe(model=model, pool=True)
    layer_id = int(details[details["layer_type"].astype(str).str.contains("dense", case=False)].iloc[0]["layer_id"])
    with pytest.raises(ValueError, match="RandomState|default_rng|Generator"):
        watcher.analyze_traps(model=model, layers=[layer_id], rng=np.random.default_rng(123), plot=False, savefig=False)


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_analyze_traps_accepts_randomstate_and_int_seed():
    model = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(8))
    watcher = WeightWatcher(model=model)
    details = watcher.describe(model=model, pool=True)
    layer_id = int(details[details["layer_type"].astype(str).str.contains("dense", case=False)].iloc[0]["layer_id"])

    df_seed = watcher.analyze_traps(model=model, layers=[layer_id], rng=123, plot=False, savefig=False)
    df_state = watcher.analyze_traps(model=model, layers=[layer_id], rng=np.random.RandomState(123), plot=False, savefig=False)
    assert list(df_seed["perm_mode_index"]) == list(df_state["perm_mode_index"])


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_replace_layer_weights_preserves_dtype_device_directly():
    layer = torch.nn.Linear(8, 8, bias=True).to(dtype=torch.float32, device="cpu")
    wrapped = PyTorchLayer(layer, layer_id=0, name="linear")

    orig_weight_dtype = layer.weight.dtype
    orig_weight_device = layer.weight.device
    orig_bias_dtype = layer.bias.dtype
    orig_bias_device = layer.bias.device

    W64 = np.random.default_rng(0).normal(size=(8, 8)).astype(np.float64)
    B64 = np.random.default_rng(1).normal(size=(8,)).astype(np.float64)
    wrapped.replace_layer_weights(W64, B64)

    assert layer.weight.dtype == orig_weight_dtype == torch.float32
    assert layer.weight.device == orig_weight_device
    assert layer.bias.dtype == orig_bias_dtype == torch.float32
    assert layer.bias.device == orig_bias_device


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_remove_traps_preserves_pytorch_layer_dtype_and_forward_pass():
    rng = np.random.default_rng(55)
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(96, 96, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(96, 10, bias=True),
    ).to(dtype=torch.float32)

    first_linear = model[1]
    orig_weight_dtype = first_linear.weight.dtype
    orig_weight_device = first_linear.weight.device
    orig_bias_dtype = first_linear.bias.dtype
    orig_bias_device = first_linear.bias.device

    W_base = rng.normal(0.0, 0.03, size=(96, 96)).astype(np.float32)
    u = np.zeros(96, dtype=np.float32)
    v = np.zeros(96, dtype=np.float32)
    u[3] = 1.0
    v[77] = 1.0
    T1 = (10.0 * np.outer(u, v)).astype(np.float32)
    W = (W_base + T1).astype(np.float32)

    with torch.no_grad():
        first_linear.weight.copy_(torch.from_numpy(W))
        first_linear.bias.zero_()

    watcher = WeightWatcher(model=model)
    details = watcher.describe(model=model, pool=True)
    dense_rows = details[details["layer_type"].astype(str).str.contains("dense", case=False)]
    target_layer_id = int(dense_rows.iloc[0]["layer_id"])

    randomized_model = watcher.randomize_model(model=model, layers=[target_layer_id], rng=99, pool=True)
    watcher.remove_traps(randomized_model=randomized_model, layers=[target_layer_id], trap_indices=[1], seed=99, pool=True, plot=False)

    assert first_linear.weight.dtype == orig_weight_dtype == torch.float32
    assert first_linear.weight.device == orig_weight_device
    assert first_linear.bias.dtype == orig_bias_dtype == torch.float32
    assert first_linear.bias.device == orig_bias_device

    x = torch.randn(4, 96, dtype=torch.float32, device=orig_weight_device)
    y = model(x)
    assert y.shape == (4, 10)
    assert y.dtype == torch.float32

    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        model_mps = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(96, 96, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(96, 10, bias=True),
        ).to(device=mps_device, dtype=torch.float32)

        with torch.no_grad():
            model_mps[1].weight.copy_(torch.from_numpy(W).to(device=mps_device, dtype=torch.float32))
            model_mps[1].bias.zero_()

        watcher_mps = WeightWatcher(model=model_mps)
        details_mps = watcher_mps.describe(model=model_mps, pool=True)
        dense_rows_mps = details_mps[details_mps["layer_type"].astype(str).str.contains("dense", case=False)]
        target_layer_id_mps = int(dense_rows_mps.iloc[0]["layer_id"])
        randomized_model_mps = watcher_mps.randomize_model(model=model_mps, layers=[target_layer_id_mps], rng=99, pool=True)
        watcher_mps.remove_traps(randomized_model=randomized_model_mps, layers=[target_layer_id_mps], trap_indices=[1], seed=99, pool=True, plot=False)

        assert model_mps[1].weight.dtype == torch.float32
        assert model_mps[1].weight.device.type == "mps"
        x_mps = torch.randn(4, 96, dtype=torch.float32, device=mps_device)
        y_mps = model_mps(x_mps)
        assert y_mps.shape == (4, 10)
