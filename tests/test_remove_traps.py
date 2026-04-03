import numpy as np
import pytest

from weightwatcher.constants import CHANNELS, FRAMEWORK, LAYER_TYPE, DEFAULT_PARAMS
from weightwatcher.RMT_Util import svd_full
from weightwatcher.weightwatcher import FrameworkLayer, WWLayer, WeightWatcher


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
    with pytest.raises(ValueError):
        watcher.apply_remove_traps(ww_bad, trap_indices=[1, 2, 3], params=make_test_params(), seed=12)

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
        model={"dummy_weight": np.array([1.0])},
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
