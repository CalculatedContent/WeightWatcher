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

        df_top10 = self.watcher.analyze_traps(plot=False, savefig=False, trap_burden=True, trap_burden_variant="top10")
        self.assertIn("trap_variance_burden", df_top10.columns)
        self.assertIn("permute_fingerprint", df_top10.columns)

    def test_analyze_traps_trap_burden_invalid_variant(self):
        with self.assertRaises(ValueError):
            self.watcher.analyze_traps(plot=False, savefig=False, trap_burden=True, trap_burden_variant="bad")

    def test_pr359_old_metrics_exist(self):
        df = self.watcher.analyze_traps(plot=False, savefig=False, rng=1337)
        required = {
            "trap_delta", "trap_ipr", "trap_q", "trap_diffuseness",
            "trap_q_uniform", "trap_diffuseness_uniform", "trap_top_sector_overlap",
            "trap_variance_burden_old", "layer_trap_variance_burden",
            "top_sector_l", "top_sector_l_effective",
        }
        self.assertTrue(required.issubset(set(df.columns)))

    def test_pr359_old_formula_rowwise_and_layer_aggregate(self):
        df = self.watcher.analyze_traps(plot=False, savefig=False, rng=1337)
        if len(df) == 0:
            self.skipTest("No traps detected in this environment")
        mask = np.isfinite(df["trap_delta"]) & np.isfinite(df["trap_q"]) & np.isfinite(df["trap_top_sector_overlap"]) & np.isfinite(df["trap_variance_burden_old"])
        if mask.any():
            expected = (df.loc[mask, "trap_delta"] ** 2) * df.loc[mask, "trap_q"] * (df.loc[mask, "trap_top_sector_overlap"] ** 2)
            self.assertTrue(np.allclose(df.loc[mask, "trap_variance_burden_old"], expected))
        for _, g in df.groupby("layer_id"):
            layer_val = g["layer_trap_variance_burden"].iloc[0]
            expected_layer = np.nansum(g["trap_variance_burden_old"].to_numpy())
            self.assertTrue(np.isclose(layer_val, expected_layer, equal_nan=True))

    def test_top_sector_l_argument(self):
        df1 = self.watcher.analyze_traps(plot=False, savefig=False, rng=1337, top_sector_l=1)
        df2 = self.watcher.analyze_traps(plot=False, savefig=False, rng=1337, top_sector_l=2)
        if len(df1) > 0:
            self.assertTrue((df1["top_sector_l"] == 1).all())
            self.assertTrue(((df1["top_sector_l_effective"] >= 1) & (df1["top_sector_l_effective"] <= 1)).all())
        if len(df2) > 0:
            self.assertTrue((df2["top_sector_l"] == 2).all())
            self.assertTrue(((df2["top_sector_l_effective"] >= 1) & (df2["top_sector_l_effective"] <= 2)).all())

    def test_old_and_new_burdens_coexist(self):
        df = self.watcher.analyze_traps(plot=False, savefig=False, trap_burden=True, trap_burden_variant="top5")
        for col in ["trap_variance_burden_old", "trap_variance_burden_ipr", "trap_variance_burden_top5", "trap_variance_burden"]:
            self.assertIn(col, df.columns)

    def test_no_trap_fft_api_or_columns(self):
        with self.assertRaises(TypeError):
            self.watcher.analyze_traps(plot=False, savefig=False, trap_fft=True)
        df = self.watcher.analyze_traps(plot=False, savefig=False)
        self.assertFalse(any(c.startswith("trap_fft") for c in df.columns))
        self.assertFalse(any(c.startswith("trap_variance_burden__") for c in df.columns))



    def test_analyze_traps_public_trap_indices_are_1_based(self):
        df, trap_state = self.watcher.analyze_traps(plot=False, savefig=False, return_artifacts=True)
        if len(df) == 0:
            self.skipTest("No traps detected in this environment")
        self.assertGreaterEqual(int(df["trap_index"].min()), 1)
        for _, g in df.groupby("layer_id"):
            vals = sorted(g["trap_index"].astype(int).tolist())
            self.assertEqual(vals, list(range(1, len(vals) + 1)))
        for lid, layer_state in trap_state.get("layers", {}).items():
            arts = layer_state.get("artifacts", [])
            if not arts:
                continue
            self.assertEqual([int(a["trap_index"]) for a in arts], list(range(1, len(arts) + 1)))

    def test_analyze_traps_fast_mode_skips_original_basis(self):
        from unittest.mock import patch
        with patch.object(ww.WeightWatcher, "compute_original_basis_for_traps", side_effect=AssertionError("should not call")):
            df = self.watcher.analyze_traps(plot=False, savefig=False, trap_burden=True, trap_burden_mode="fast")
        self.assertIsInstance(df, pd.DataFrame)

    def test_analyze_traps_fast_mode_skips_full_bulk_reference(self):
        from unittest.mock import patch
        with patch.object(ww.WeightWatcher, "compute_bulk_trap_reference_metrics", side_effect=AssertionError("should not call")):
            df = self.watcher.analyze_traps(plot=False, savefig=False, trap_burden=True, trap_burden_mode="fast")
        self.assertIn("B_absDelta_ipr_ovlamvar", df.columns)

    def test_analyze_traps_rejects_model_and_randomized_model_together(self):
        with self.assertRaises(ValueError):
            self.watcher.analyze_traps(model=self.model, randomized_model=self.model, plot=False, savefig=False)


if __name__ == "__main__":
    unittest.main()
