import unittest

import weightwatcher as ww


class TestTrapDiffuseness(unittest.TestCase):

    def setUp(self):
        self.watcher = ww.WeightWatcher()

    def test_assess_trap_diffuseness_returns_expected_keys(self):
        trap = {
            "u_length": 20,
            "v_length": 20,
            "u_effective_support": 10,
            "v_effective_support": 11,
            "u_squared_amp_entropy": 2.0,
            "v_squared_amp_entropy": 2.1,
            "u_top1_mass": 0.10,
            "v_top1_mass": 0.11,
            "u_gini_abs": 0.20,
            "v_gini_abs": 0.25,
            "left_top_mass": 0.12,
            "right_top_mass": 0.13,
            "trap_eval_minus_bulk": 0.5,
            "mp_bulk_max": 1.0,
        }
        out = self.watcher.assess_trap_diffuseness(trap)
        self.assertIn("trap_diffuseness_score", out)
        self.assertIn("trap_risk_score", out)
        self.assertIn("trap_assessment", out)

    def test_assess_trap_diffuseness_localized_vs_diffuse(self):
        localized = {
            "u_length": 20,
            "v_length": 20,
            "u_effective_support": 1.5,
            "v_effective_support": 1.8,
            "u_squared_amp_entropy": 0.2,
            "v_squared_amp_entropy": 0.3,
            "u_top1_mass": 0.85,
            "v_top1_mass": 0.80,
            "u_gini_abs": 0.92,
            "v_gini_abs": 0.90,
            "left_top_mass": 0.88,
            "right_top_mass": 0.86,
            "trap_eval_minus_bulk": 4.0,
            "mp_bulk_max": 1.0,
        }

        diffuse = {
            "u_length": 20,
            "v_length": 20,
            "u_effective_support": 15.0,
            "v_effective_support": 16.0,
            "u_squared_amp_entropy": 2.7,
            "v_squared_amp_entropy": 2.8,
            "u_top1_mass": 0.10,
            "v_top1_mass": 0.10,
            "u_gini_abs": 0.20,
            "v_gini_abs": 0.22,
            "left_top_mass": 0.15,
            "right_top_mass": 0.14,
            "trap_eval_minus_bulk": 0.2,
            "mp_bulk_max": 1.0,
        }

        loc_out = self.watcher.assess_trap_diffuseness(localized)
        dif_out = self.watcher.assess_trap_diffuseness(diffuse)

        self.assertLess(loc_out["trap_diffuseness_score"], dif_out["trap_diffuseness_score"])
        self.assertGreaterEqual(loc_out["trap_risk_score"], dif_out["trap_risk_score"])

    def test_risk_score_decreases_monotonically_with_diffuseness_for_fixed_strength(self):
        base = {
            "u_length": 20,
            "v_length": 20,
            "trap_eval_minus_bulk": 0.8,
            "mp_bulk_max": 1.0,
        }
        low_diff = dict(base, **{
            "u_effective_support": 1.0,
            "v_effective_support": 1.0,
            "u_squared_amp_entropy": 0.1,
            "v_squared_amp_entropy": 0.1,
            "u_top1_mass": 0.95,
            "v_top1_mass": 0.95,
            "u_gini_abs": 0.95,
            "v_gini_abs": 0.95,
            "left_top_mass": 0.95,
            "right_top_mass": 0.95,
        })
        mid_diff = dict(base, **{
            "u_effective_support": 10.0,
            "v_effective_support": 10.0,
            "u_squared_amp_entropy": 1.5,
            "v_squared_amp_entropy": 1.5,
            "u_top1_mass": 0.4,
            "v_top1_mass": 0.4,
            "u_gini_abs": 0.6,
            "v_gini_abs": 0.6,
            "left_top_mass": 0.4,
            "right_top_mass": 0.4,
        })
        high_diff = dict(base, **{
            "u_effective_support": 20.0,
            "v_effective_support": 20.0,
            "u_squared_amp_entropy": 3.0,
            "v_squared_amp_entropy": 3.0,
            "u_top1_mass": 0.0,
            "v_top1_mass": 0.0,
            "u_gini_abs": 0.0,
            "v_gini_abs": 0.0,
            "left_top_mass": 0.0,
            "right_top_mass": 0.0,
        })

        out_low = self.watcher.assess_trap_diffuseness(low_diff)
        out_mid = self.watcher.assess_trap_diffuseness(mid_diff)
        out_high = self.watcher.assess_trap_diffuseness(high_diff)

        self.assertLessEqual(out_low["trap_diffuseness_score"], out_mid["trap_diffuseness_score"])
        self.assertLessEqual(out_mid["trap_diffuseness_score"], out_high["trap_diffuseness_score"])
        self.assertGreaterEqual(out_low["trap_risk_score"], out_mid["trap_risk_score"])
        self.assertGreaterEqual(out_mid["trap_risk_score"], out_high["trap_risk_score"])

    def test_risk_score_is_zero_when_diffuseness_is_one(self):
        trap = {
            "u_length": 20,
            "v_length": 20,
            "u_effective_support": 20.0,
            "v_effective_support": 20.0,
            "u_squared_amp_entropy": 3.0,
            "v_squared_amp_entropy": 3.0,
            "u_top1_mass": 0.0,
            "v_top1_mass": 0.0,
            "u_gini_abs": 0.0,
            "v_gini_abs": 0.0,
            "left_top_mass": 0.0,
            "right_top_mass": 0.0,
            "trap_eval_minus_bulk": 999.0,
            "mp_bulk_max": 1.0,
        }
        out = self.watcher.assess_trap_diffuseness(trap)
        self.assertEqual(out["trap_diffuseness_score"], 1.0)
        self.assertEqual(out["trap_risk_score"], 0.0)
        self.assertEqual(out["trap_assessment"], "benign_diffuse")

    def test_risk_score_matches_base_strength_when_diffuseness_is_zero(self):
        trap = {
            "u_length": 20,
            "v_length": 20,
            "u_effective_support": 0.0,
            "v_effective_support": 0.0,
            "u_squared_amp_entropy": 0.0,
            "v_squared_amp_entropy": 0.0,
            "u_top1_mass": 1.0,
            "v_top1_mass": 1.0,
            "u_gini_abs": 1.0,
            "v_gini_abs": 1.0,
            "left_top_mass": 1.0,
            "right_top_mass": 1.0,
            "trap_eval_minus_bulk": 0.3,
            "mp_bulk_max": 2.0,
        }
        out = self.watcher.assess_trap_diffuseness(trap)
        self.assertEqual(out["trap_diffuseness_score"], 0.0)
        self.assertAlmostEqual(out["trap_risk_score"], 0.15, places=8)

        trap_hi = dict(trap, **{"trap_eval_minus_bulk": 5.0, "mp_bulk_max": 1.0})
        out_hi = self.watcher.assess_trap_diffuseness(trap_hi)
        self.assertEqual(out_hi["trap_diffuseness_score"], 0.0)
        self.assertEqual(out_hi["trap_risk_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
