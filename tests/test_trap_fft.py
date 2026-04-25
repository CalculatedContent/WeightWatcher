import unittest
from unittest.mock import patch
import numpy as np

import weightwatcher as ww
import weightwatcher.trap_fourier as tf
import weightwatcher.trap_burden_variants as tbv

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class TestTrapFourierHelpers(unittest.TestCase):
    def test_mass_sums_to_one(self):
        rng = np.random.RandomState(123)
        v = rng.normal(size=31)
        raw = tf.fourier_mass(v, fold_conjugates=False)
        self.assertAlmostEqual(float(np.sum(raw["mass"])), 1.0, places=7)

        folded = tf.fourier_mass(v, fold_conjugates=True)
        self.assertAlmostEqual(float(np.sum(folded["folded_mass"])), 1.0, places=7)

    def test_one_hot_is_fourier_delocalized(self):
        n = 32
        v = np.zeros(n)
        v[0] = 1.0
        ipr_fft, q_fft = tf.fourier_uniform_centered_q(v, fold_conjugates=False)
        self.assertAlmostEqual(ipr_fft, 1.0 / n, places=6)
        self.assertAlmostEqual(q_fft, 0.0, places=6)

    def test_sinusoid_is_fourier_localized(self):
        n = 64
        k = 7
        j = np.arange(n)
        v = np.cos(2.0 * np.pi * k * j / n)
        v /= np.linalg.norm(v)

        top_mass, idx, _ = tf.fourier_top_frequency_mass(v, top_frequency_l=1, fold_conjugates=True)
        self.assertGreater(top_mass, 0.9)
        self.assertTrue(len(idx) > 0)

        _, q_fft = tf.fourier_uniform_centered_q(v, fold_conjugates=True)
        self.assertGreater(q_fft, 0.5)

    def test_default_variants_have_fft_only_when_enabled(self):
        base = tbv.resolve_burden_variant_configs("default", trap_fft=False)
        self.assertFalse(any(cfg["name"].startswith("fft_") for cfg in base))

        with_fft = tbv.resolve_burden_variant_configs("default", trap_fft=True)
        self.assertTrue(any(cfg["name"].startswith("fft_") for cfg in with_fft))


if TORCH_AVAILABLE:
    class TinyTrapNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 12, bias=False)
            self.fc2 = nn.Linear(12, 10, bias=False)
            with torch.no_grad():
                u = torch.linspace(1.0, 2.0, steps=12)
                v = torch.linspace(-2.0, 1.0, steps=16)
                self.fc1.weight.copy_(35.0 * torch.outer(u, v))

                u2 = torch.linspace(1.0, 1.5, steps=10)
                v2 = torch.linspace(-1.0, 2.0, steps=12)
                self.fc2.weight.copy_(20.0 * torch.outer(u2, v2))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for trap FFT integration tests")
class TestTrapFFTAnalyzeTrapsIntegration(unittest.TestCase):
    def setUp(self):
        self.watcher = ww.WeightWatcher(model=TinyTrapNet())

    def test_trap_fft_true_does_not_call_matrix_fft(self):
        with patch.object(self.watcher, "apply_FFT", side_effect=RuntimeError("apply_FFT should not be called")):
            df = self.watcher.analyze_traps(
                plot=False,
                savefig=False,
                rng=1337,
                fft=False,
                trap_fft=True,
            )
        self.assertIsNotNone(df)

    def test_trap_fft_columns_present(self):
        df = self.watcher.analyze_traps(
            plot=False,
            savefig=False,
            rng=1337,
            fft=False,
            trap_fft=True,
            burden_variants="default",
        )
        required = {
            "trap_fft_ipr_right_perm",
            "trap_fft_q_uniform_right_perm",
            "trap_fft_q_pt_right_perm",
            "trap_fft_top_frequency_mass_right_perm",
            "trap_fft_peak_frequency_right_perm",
            "trap_fft_peak_mass_right_perm",
            "trap_variance_burden__fft_uniform_lr_geom_fft_topmass",
        }
        self.assertTrue(required.issubset(set(df.columns)))

    def test_fft_true_preserves_old_matrix_fft_behavior(self):
        called = {"n": 0}

        def _counting_fft(*args, **kwargs):
            called["n"] += 1
            return ww.WeightWatcher.apply_FFT(self.watcher, *args, **kwargs)

        with patch.object(self.watcher, "apply_FFT", side_effect=_counting_fft):
            self.watcher.analyze_traps(
                plot=False,
                savefig=False,
                rng=1337,
                fft=True,
                trap_fft=False,
            )

        self.assertGreater(called["n"], 0)


if __name__ == "__main__":
    unittest.main()
