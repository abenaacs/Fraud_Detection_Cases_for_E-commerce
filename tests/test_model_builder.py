import unittest
import pandas as pd
from src.model_builder import ModelBuilder


class TestModelBuilder(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataset
        self.data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [5, 4, 3, 2, 1],
                "target": [0, 1, 0, 1, 0],
            }
        )
        self.data.to_csv("test_data.csv", index=False)
        self.builder = ModelBuilder(data_path="test_data.csv", target_column="target")

    def test_split_data(self):
        self.builder.split_data(test_size=0.2, random_state=42)
        self.assertEqual(
            len(self.builder.X_train) + len(self.builder.X_test), len(self.builder.X)
        )

    def test_train_and_evaluate(self):
        self.builder.split_data(test_size=0.2, random_state=42)
        results = self.builder.train_and_evaluate()
        self.assertIn("Logistic Regression", results)
        self.assertGreater(results["Logistic Regression"]["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
