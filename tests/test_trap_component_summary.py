import numpy as np

from weightwatcher.trap_analysis import _top_trap_component_row


def test_top_trap_component_row_extracts_top_10_weight_and_coeff_pairs():
    trap = np.array(
        [
            [0.1, -0.9, 0.3],
            [0.5, 0.2, -0.4],
        ]
    )
    weights = np.array(
        [
            [11.0, 12.0, 13.0],
            [21.0, 22.0, 23.0],
        ]
    )
    row = {
        "layer_id": 7,
        "name": "dense",
        "trap_index": 1,
        "trap_assessment": "mixed",
        "trap_risk_score": 0.4,
        "T_orig": trap,
    }

    out = _top_trap_component_row(row=row, weight_matrix=weights, top_k=3)

    assert out["layer_id"] == 7
    assert out["trap_index"] == 1
    assert np.isclose(out["Wij_1"], 12.0)  # |coeff| = 0.9, largest
    assert np.isclose(out["Cij_1"], -1.0)  # normalized by max |coeff|
    assert np.isclose(out["Wij_2"], 21.0)  # |coeff| = 0.5
    assert np.isclose(out["Cij_2"], 0.5 / 0.9)
