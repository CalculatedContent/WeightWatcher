import unittest
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

import weightwatcher as ww
import weightwatcher.trap_burden_variants as tbv


class TestTrapBurdenVariantMath(unittest.TestCase):
    def test_current_pr358_variant_formula(self):
        components = {
            "trap_spectral_edge_ratio_current": 0.2,
            "trap_q_pt_right_perm": 0.5,
            "trap_top_sector_overlap_right": 0.3,
        }
        cfg = [c for c in tbv.DEFAULT_BURDEN_VARIANTS if c["name"] == "current_pr358"][0]
        v = tbv.compute_burden_variant(components, cfg)
        self.assertAlmostEqual(v, 0.2 ** 2 * 0.5 * 0.3 ** 2)

    def test_uniform_localization_basics(self):
        v = np.ones(10) / np.sqrt(10)
        ipr, q = tbv.localization_uniform_centered(v)
        self.assertAlmostEqual(ipr, 0.1)
        self.assertAlmostEqual(q, 0.0)

        w = np.zeros(10)
        w[0] = 1.0
        ipr, q = tbv.localization_uniform_centered(w)
        self.assertAlmostEqual(ipr, 1.0)
        self.assertAlmostEqual(q, 1.0)

    def test_porter_thomas_localization(self):
        # n=10 -> expected real PT IPR is 3/(10+2)=0.25
        # vector with 4 equal non-zero entries has IPR=0.25 exactly
        v = np.array([0.5, 0.5, 0.5, 0.5] + [0.0] * 6)
        ipr, q = tbv.localization_porter_thomas_centered(v, beta="real")
        self.assertAlmostEqual(ipr, 3.0 / 12.0)
        self.assertAlmostEqual(q, 0.0, places=7)

        w = np.zeros(10)
        w[0] = 1.0
        _, q1 = tbv.localization_porter_thomas_centered(w, beta="real")
        self.assertTrue(q1 <= 1.0 and q1 >= 0.0)
        self.assertAlmostEqual(q1, 1.0)

    def test_spectral_modes(self):
        eval_perm = 12.0
        mp_bulk_max = 10.0
        total = 100.0
        self.assertAlmostEqual(
            tbv.spectral_excess(eval_perm, mp_bulk_max, total, mode="edge_ratio_current"),
            0.2,
        )
        self.assertAlmostEqual(
            tbv.spectral_excess(eval_perm, mp_bulk_max, total, mode="total_excess"),
            0.02,
        )
        self.assertAlmostEqual(
            tbv.spectral_excess(eval_perm, mp_bulk_max, total, mode="total_fraction"),
            0.12,
        )

    def test_combine_lr(self):
        self.assertAlmostEqual(tbv.combine_lr(0.2, 0.8, "geom"), 0.4)
        self.assertAlmostEqual(tbv.combine_lr(0.2, 0.8, "min"), 0.2)
        self.assertAlmostEqual(tbv.combine_lr(0.2, 0.8, "max"), 0.8)
        self.assertAlmostEqual(tbv.combine_lr(0.2, 0.8, "mean"), 0.5)
        self.assertAlmostEqual(tbv.combine_lr(0.2, 0.8, "product"), 0.16)
        self.assertTrue(np.isnan(tbv.combine_lr(np.nan, 0.8, "mean")))


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

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for analyze_traps variant tests")
class TestTrapBurdenVariantAPI(unittest.TestCase):
    def setUp(self):
        self.watcher = ww.WeightWatcher(model=TinyTrapNet())

    def test_analyze_traps_default_has_no_variant_cols(self):
        df = self.watcher.analyze_traps(plot=False, savefig=False, rng=1337)
        self.assertFalse(any(c.startswith("trap_variance_burden__") for c in df.columns))

    def test_analyze_traps_default_variant_sweep_columns(self):
        df = self.watcher.analyze_traps(
            plot=False,
            savefig=False,
            rng=1337,
            burden_variants="default",
        )
        variant_cols = [c for c in df.columns if c.startswith("trap_variance_burden__")]
        self.assertTrue(len(variant_cols) > 0)
        self.assertIn("trap_variance_burden__current_pr358", variant_cols)

    def test_analyze_traps_component_columns(self):
        df = self.watcher.analyze_traps(
            plot=False,
            savefig=False,
            rng=1337,
            burden_variants="default",
            return_burden_components=True,
        )
        required = {
            "trap_q_pt_left_perm",
            "trap_q_pt_right_perm",
            "trap_q_pt_perm_lr_geom",
            "trap_top_sector_overlap_left",
            "trap_top_sector_overlap_right",
            "trap_spectral_total_excess",
            "trap_perm_total_variance",
        }
        self.assertTrue(required.issubset(set(df.columns)))

    def test_analyze_traps_raw_columns_control(self):
        df = self.watcher.analyze_traps(
            plot=False,
            savefig=False,
            rng=1337,
            burden_variants="default",
            return_burden_raw=False,
        )
        for c in ["u_perm", "v_perm", "u_trap", "v_trap", "left_overlaps", "right_overlaps", "perm_evals_sorted"]:
            self.assertNotIn(c, df.columns)

        df_raw = self.watcher.analyze_traps(
            plot=False,
            savefig=False,
            rng=1337,
            burden_variants="default",
            return_burden_raw=True,
        )
        if len(df_raw) == 0:
            self.skipTest("No traps detected in this environment")
        for c in ["u_perm", "v_perm", "u_trap", "v_trap", "left_overlaps", "right_overlaps", "perm_evals_sorted"]:
            self.assertIn(c, df_raw.columns)

    def test_current_variant_matches_base_burden(self):
        df = self.watcher.analyze_traps(
            plot=False,
            savefig=False,
            rng=1337,
            burden_variants="default",
            return_burden_components=True,
        )
        if len(df) == 0:
            self.skipTest("No traps detected in this environment")
        self.assertTrue(np.allclose(
            df["trap_variance_burden"],
            df["trap_variance_burden__current_pr358"],
            equal_nan=True,
        ))


if __name__ == "__main__":
    unittest.main()
