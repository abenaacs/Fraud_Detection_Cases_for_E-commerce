import unittest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.explainability import Explainability


class TestExplainability(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataset
        self.X_train = pd.DataFrame(
            {"feature1": [1, 2, 3, 4], "feature2": [4, 3, 2, 1]}
        )
        self.X_test = pd.DataFrame({"feature1": [5], "feature2": [1]})
        self.model = RandomForestClassifier().fit(self.X_train, [0, 1, 0, 1])

    def test_explain_with_shap(self):
        explainer = Explainability(
            model=self.model,
            X_train=self.X_train,
            X_test=self.X_test,
            feature_names=self.X_train.columns,
        )
        try:
            explainer.explain_with_shap()
            self.assertTrue(True)  # No exceptions raised
        except Exception as e:
            self.fail(f"explain_with_shap raised an exception: {e}")

    def test_explain_with_lime(self):
        explainer = Explainability(
            model=self.model,
            X_train=self.X_train,
            X_test=self.X_test,
            feature_names=self.X_train.columns,
        )
        try:
            explainer.explain_with_lime()
            self.assertTrue(True)  # No exceptions raised
        except Exception as e:
            self.fail(f"explain_with_lime raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
