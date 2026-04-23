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


class TestTrapMetricHelpers(unittest.TestCase):

    def setUp(self):
        self.watcher = ww.WeightWatcher()

    def test_compute_trap_delta(self):
        self.assertAlmostEqual(self.watcher.compute_trap_delta(12.0, 10.0), 0.2)
        self.assertAlmostEqual(self.watcher.compute_trap_delta(8.0, 10.0), 0.0)
        self.assertTrue(np.isnan(self.watcher.compute_trap_delta(12.0, 0.0)))

    def test_compute_trap_ipr_q_porter_thomas_baseline_vector(self):
        # m=10 => E_PT[IPR]=3/(m+2)=0.25
        v = np.array([0.5, 0.5, 0.5, 0.5] + [0.0] * 6)
        ipr, q = self.watcher.compute_trap_ipr_q(v)
        self.assertAlmostEqual(ipr, 0.25)
        self.assertAlmostEqual(q, 0.0, places=7)

    def test_compute_trap_ipr_q_uniform_vector_clips_to_zero_under_pt(self):
        v = np.ones(10) / np.sqrt(10)
        ipr, q = self.watcher.compute_trap_ipr_q(v)
        self.assertAlmostEqual(ipr, 0.1)
        self.assertAlmostEqual(q, 0.0)

    def test_compute_trap_ipr_q_one_hot_vector(self):
        v = np.zeros(10)
        v[0] = 1.0
        ipr, q = self.watcher.compute_trap_ipr_q(v)
        self.assertAlmostEqual(ipr, 1.0)
        self.assertAlmostEqual(q, 1.0)

    def test_compute_trap_ipr_q_uniform_legacy_field_behavior(self):
        v = np.ones(10) / np.sqrt(10)
        ipr, q = self.watcher.compute_trap_ipr_q_uniform(v)
        self.assertAlmostEqual(ipr, 0.1)
        self.assertAlmostEqual(q, 0.0)

        onehot = np.zeros(10)
        onehot[0] = 1.0
        ipr, q = self.watcher.compute_trap_ipr_q_uniform(onehot)
        self.assertAlmostEqual(ipr, 1.0)
        self.assertAlmostEqual(q, 1.0)

    def test_compute_top_sector_overlap(self):
        overlaps = np.array([0.25, 0.10, 0.05, 0.60])

        overlap_1, ell_eff = self.watcher.compute_top_sector_overlap(overlaps, 1)
        self.assertAlmostEqual(overlap_1, 0.25)
        self.assertEqual(ell_eff, 1)

        overlap_2, ell_eff = self.watcher.compute_top_sector_overlap(overlaps, 2)
        self.assertAlmostEqual(overlap_2, 0.35)
        self.assertEqual(ell_eff, 2)

    def test_compute_trap_variance_burden(self):
        burden = self.watcher.compute_trap_variance_burden(
            trap_delta=0.2,
            trap_q=0.5,
            trap_top_sector_overlap=0.3,
        )
        self.assertAlmostEqual(burden, 0.2**2 * 0.5 * 0.3**2)


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
            "sigma_perm", "mp_bulk_max", "left_top_mass", "right_top_mass"
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

    def test_analyze_traps_contains_paper_metric_columns(self):
        df = self.watcher.analyze_traps(plot=False, savefig=False, rng=1337)
        required = {
            "top_sector_l",
            "top_sector_l_effective",
            "trap_delta",
            "trap_ipr",
            "trap_q",
            "trap_diffuseness",
            "trap_q_uniform",
            "trap_diffuseness_uniform",
            "trap_top_sector_overlap",
            "trap_variance_burden",
            "layer_trap_variance_burden",
        }
        self.assertTrue(required.issubset(set(df.columns)))

    def test_analyze_traps_respects_top_sector_l_argument(self):
        df1 = self.watcher.analyze_traps(plot=False, savefig=False, rng=1337, top_sector_l=1)
        df2 = self.watcher.analyze_traps(plot=False, savefig=False, rng=1337, top_sector_l=2)

        if len(df1) == 0 or len(df2) == 0:
            self.skipTest("No traps detected in this environment")

        self.assertTrue((df1["top_sector_l"] == 1).all())
        self.assertTrue((df2["top_sector_l"] == 2).all())
        self.assertTrue((df1["top_sector_l_effective"] >= 1).all())
        self.assertTrue((df1["top_sector_l_effective"] <= df1["top_sector_l"]).all())
        self.assertTrue((df2["top_sector_l_effective"] >= 1).all())
        self.assertTrue((df2["top_sector_l_effective"] <= df2["top_sector_l"]).all())

    def test_trap_variance_burden_formula_rowwise(self):
        df = self.watcher.analyze_traps(plot=False, savefig=False, rng=1337)
        if len(df) == 0:
            self.skipTest("No traps detected in this environment")

        for _, row in df.iterrows():
            components = [row["trap_delta"], row["trap_q"], row["trap_top_sector_overlap"]]
            if not np.all(np.isfinite(components)):
                continue
            expected = (row["trap_delta"] ** 2) * row["trap_q"] * (row["trap_top_sector_overlap"] ** 2)
            self.assertAlmostEqual(row["trap_variance_burden"], expected)

    def test_layer_trap_variance_burden_aggregate(self):
        df = self.watcher.analyze_traps(plot=False, savefig=False, rng=1337)
        if len(df) == 0:
            self.skipTest("No traps detected in this environment")

        for layer_id, subdf in df.groupby("layer_id"):
            expected = subdf["trap_variance_burden"].sum()
            observed = subdf["layer_trap_variance_burden"].iloc[0]
            self.assertAlmostEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
