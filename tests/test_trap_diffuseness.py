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


if __name__ == "__main__":
    unittest.main()
