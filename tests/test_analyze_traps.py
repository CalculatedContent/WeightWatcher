import unittest
import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

import weightwatcher as ww


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


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for analyze_traps tests")
class TestAnalyzeTraps(unittest.TestCase):

    def setUp(self):
        self.model = TinyTrapNet()
        self.watcher = ww.WeightWatcher(model=self.model)

    def test_analyze_traps_method_exists(self):
        self.assertTrue(hasattr(self.watcher, "analyze_traps"))

    def test_analyze_traps_returns_dataframe(self):
        np.random.seed(123)
        df = self.watcher.analyze_traps(plot=False, savefig=False)
        self.assertIsInstance(df, pd.DataFrame)

    def test_analyze_traps_columns(self):
        np.random.seed(123)
        df = self.watcher.analyze_traps(plot=False, savefig=False)
        expected_cols = {
            "layer_id", "name", "trap_index", "perm_mode_index",
            "sigma_perm", "mp_bulk_max", "left_top_mass", "right_top_mass",
            "top_5_mass", "top_10_mass",
            "bulk_localization_mean", "bulk_localization_std",
            "bulk_top_5_mass_mean", "bulk_top_5_mass_std",
            "bulk_top_10_mass_mean", "bulk_top_10_mass_std",
        }
        self.assertTrue(expected_cols.issubset(set(df.columns)))

    def test_analyze_traps_no_powerlaw_columns_required(self):
        np.random.seed(123)
        df = self.watcher.analyze_traps(plot=False, savefig=False)
        self.assertNotIn("alpha", df.columns)
        self.assertNotIn("xmin", df.columns)
        self.assertNotIn("xmax", df.columns)

    def test_analyze_traps_reproducible_when_seed_fixed(self):
        np.random.seed(999)
        df1 = self.watcher.analyze_traps(plot=False, savefig=False)
        np.random.seed(999)
        df2 = self.watcher.analyze_traps(plot=False, savefig=False)

        self.assertEqual(len(df1), len(df2))
        self.assertListEqual(df1["layer_id"].tolist(), df2["layer_id"].tolist())
        self.assertListEqual(df1["perm_mode_index"].tolist(), df2["perm_mode_index"].tolist())

    def test_analyze_traps_reproducible_with_rng_seed(self):
        df1 = self.watcher.analyze_traps(plot=False, savefig=False, rng=1337)
        df2 = self.watcher.analyze_traps(plot=False, savefig=False, rng=1337)

        self.assertEqual(len(df1), len(df2))
        self.assertListEqual(df1["layer_id"].tolist(), df2["layer_id"].tolist())
        self.assertListEqual(df1["perm_mode_index"].tolist(), df2["perm_mode_index"].tolist())

    def test_analyze_traps_respects_layer_filter(self):
        np.random.seed(123)
        all_df = self.watcher.analyze_traps(plot=False, savefig=False)
        if len(all_df) == 0:
            self.skipTest("No traps detected in this environment")

        layer_id = int(all_df["layer_id"].iloc[0])
        np.random.seed(123)
        layer_df = self.watcher.analyze_traps(layers=[layer_id], plot=False, savefig=False)
        self.assertTrue(set(layer_df["layer_id"].unique()).issubset({layer_id}))

    def test_analyze_traps_skips_ambiguous_multi_Wmat_layers_safely(self):
        conv_model = nn.Conv2d(3, 8, kernel_size=3, bias=False)
        watcher = ww.WeightWatcher(model=conv_model)

        np.random.seed(123)
        df = watcher.analyze_traps(plot=False, savefig=False, pool=True)
        self.assertIsInstance(df, pd.DataFrame)

    def test_analyze_traps_contains_vector_metric_columns(self):
        np.random.seed(123)
        df = self.watcher.analyze_traps(plot=False, savefig=False)
        required = {
            "u_entropy", "u_discrete_entropy", "u_localization_ratio", "u_participation_ratio",
            "v_entropy", "v_discrete_entropy", "v_localization_ratio", "v_participation_ratio"
        }
        self.assertTrue(required.issubset(set(df.columns)))

    def test_analyze_traps_contains_order_invariant_stat_columns(self):
        np.random.seed(123)
        df = self.watcher.analyze_traps(plot=False, savefig=False)
        required = {
            "u_l2_fourth_moment", "u_effective_support", "u_gini_abs", "u_top10_mass",
            "u_squared_amp_entropy", "u_stable_rank_surrogate",
            "v_l2_fourth_moment", "v_effective_support", "v_gini_abs", "v_top10_mass",
            "v_squared_amp_entropy", "v_stable_rank_surrogate", "trap_balance_ratio",
            "trap_diffuseness_score", "trap_risk_score", "trap_assessment"
        }
        self.assertTrue(required.issubset(set(df.columns)))

    def test_order_invariant_stats_are_finite(self):
        np.random.seed(123)
        df = self.watcher.analyze_traps(plot=False, savefig=False)
        if len(df) == 0:
            self.skipTest("No traps detected in this environment")

        row = df.iloc[0]
        for col in [
            "u_l2_fourth_moment", "u_l2_sixth_moment", "u_effective_support", "u_gini_abs",
            "u_top1_mass", "u_top5_mass", "u_top10_mass", "u_squared_amp_entropy", "u_stable_rank_surrogate",
            "v_l2_fourth_moment", "v_l2_sixth_moment", "v_effective_support", "v_gini_abs",
            "v_top1_mass", "v_top5_mass", "v_top10_mass", "v_squared_amp_entropy", "v_stable_rank_surrogate",
            "trap_balance_ratio",
        ]:
            self.assertTrue(np.isfinite(row[col]))

    def test_analyze_traps_trap_burden_backward_compat(self):
        df = self.watcher.analyze_traps(plot=False, savefig=False, trap_burden=False)
        self.assertIsInstance(df, pd.DataFrame)
        for col in ["top_5_mass", "bulk_top_5_mass_mean", "bulk_top_10_mass_mean"]:
            self.assertIn(col, df.columns)

    def test_analyze_traps_trap_burden_columns_appear(self):
        df = self.watcher.analyze_traps(plot=False, savefig=False, trap_burden=True)
        required = {
            "spectral_excess_abs", "spectral_excess_rel", "trap_ipr", "bulk_ipr_mean",
            "ipr_lift_excess_pos", "top_5_lift", "log1p_top_5_lift", "ov_lam_weighted_var",
            "ov_rank_mean", "trap_variance_burden_ipr", "trap_variance_burden_top5", "trap_variance_burden",
        }
        self.assertTrue(required.issubset(set(df.columns)))

    def test_analyze_traps_trap_burden_finite_values(self):
        df = self.watcher.analyze_traps(plot=False, savefig=False, trap_burden=True)
        if len(df) == 0:
            self.skipTest("No traps detected in this environment")
        finite_mask = np.isfinite(df["spectral_excess_abs"]) & np.isfinite(df["ov_lam_weighted_var"]) & np.isfinite(df["ov_rank_mean"]) & np.isfinite(df["trap_variance_burden"])
        self.assertTrue(finite_mask.any())

    def test_analyze_traps_trap_burden_variant_selection(self):
        df_ipr = self.watcher.analyze_traps(plot=False, savefig=False, trap_burden=True, trap_burden_variant="ipr")
        mask_ipr = np.isfinite(df_ipr["trap_variance_burden"]) & np.isfinite(df_ipr["trap_variance_burden_ipr"])
        if mask_ipr.any():
            self.assertTrue(np.allclose(df_ipr.loc[mask_ipr, "trap_variance_burden"], df_ipr.loc[mask_ipr, "trap_variance_burden_ipr"]))

        df_top5 = self.watcher.analyze_traps(plot=False, savefig=False, trap_burden=True, trap_burden_variant="top5")
        mask_top5 = np.isfinite(df_top5["trap_variance_burden"]) & np.isfinite(df_top5["trap_variance_burden_top5"])
        if mask_top5.any():
            self.assertTrue(np.allclose(df_top5.loc[mask_top5, "trap_variance_burden"], df_top5.loc[mask_top5, "trap_variance_burden_top5"]))

    def test_analyze_traps_trap_burden_invalid_variant(self):
        with self.assertRaises(ValueError):
            self.watcher.analyze_traps(plot=False, savefig=False, trap_burden=True, trap_burden_variant="bad")


if __name__ == "__main__":
    unittest.main()
