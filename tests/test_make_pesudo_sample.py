import unittest

import numpy as np

from scripts.make_pesudo_sample import (
    MODEL_PREDICTION_COLUMNS,
    calculate_model_consensus,
)


class CalculateModelConsensusTests(unittest.TestCase):
    def test_clips_each_model_before_calculating_consensus(self):
        result = calculate_model_consensus(
            ridge_predictions=[-2.0],
            lasso_predictions=[2.0],
            sigmoid_predictions=[4.0],
            max_prediction_spread=5.0,
        )

        self.assertEqual(result.loc[0, "Ridge_Prediction"], 0.0)
        self.assertEqual(result.loc[0, "Prediction_Spread"], 4.0)
        self.assertAlmostEqual(result.loc[0, "Predicted_Sand_Thickness"], 2.0)
        self.assertTrue(result.loc[0, "Model_Agreement"])

    def test_five_meter_boundary_is_inclusive(self):
        result = calculate_model_consensus(
            ridge_predictions=[0.0, 0.0],
            lasso_predictions=[5.0, 5.0001],
            sigmoid_predictions=[2.0, 2.0],
            max_prediction_spread=5.0,
        )

        self.assertTrue(result.loc[0, "Model_Agreement"])
        self.assertFalse(result.loc[1, "Model_Agreement"])

    def test_returns_auditable_prediction_columns_and_mean_label(self):
        result = calculate_model_consensus(
            ridge_predictions=[1.0, 4.0],
            lasso_predictions=[2.0, 5.0],
            sigmoid_predictions=[3.0, 6.0],
            max_prediction_spread=5.0,
        )

        for column in MODEL_PREDICTION_COLUMNS:
            self.assertIn(column, result.columns)
        np.testing.assert_allclose(
            result["Predicted_Sand_Thickness"].values, [2.0, 5.0]
        )

    def test_rejects_non_finite_predictions(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            calculate_model_consensus(
                ridge_predictions=[1.0],
                lasso_predictions=[np.nan],
                sigmoid_predictions=[2.0],
                max_prediction_spread=5.0,
            )


if __name__ == "__main__":
    unittest.main()
