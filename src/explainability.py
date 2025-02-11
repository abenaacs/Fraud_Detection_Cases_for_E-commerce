import shap
import lime
import lime.lime_tabular
import pandas as pd
import matplotlib.pyplot as plt


class Explainability:
    def __init__(self, model, X_train, X_test, feature_names):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names

    def explain_with_shap(self):
        # Initialize SHAP explainer
        explainer = shap.Explainer(self.model, self.X_train)
        shap_values = explainer(self.X_test)

        # Summary Plot
        shap.summary_plot(
            shap_values, self.X_test, feature_names=self.feature_names, show=False
        )
        plt.savefig("shap_summary_plot.png")
        plt.close()

        # Force Plot (for the first instance)
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            self.X_test.iloc[0],
            feature_names=self.feature_names,
            show=False,
            matplotlib=True,
        )
        plt.savefig("shap_force_plot.png")
        plt.close()

        # Dependence Plot (for the most important feature)
        shap.dependence_plot(
            shap_values.abs.mean(0).argsort[-1],
            shap_values.values,
            self.X_test,
            feature_names=self.feature_names,
            show=False,
        )
        plt.savefig("shap_dependence_plot.png")
        plt.close()

    def explain_with_lime(self):
        # Initialize LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.feature_names,
            class_names=["Not Fraud", "Fraud"],
            verbose=True,
            mode="classification",
        )

        # Explain a single prediction
        instance_idx = 0
        exp = explainer.explain_instance(
            self.X_test.iloc[instance_idx].values,
            self.model.predict_proba,
            num_features=len(self.feature_names),
        )
        exp.save_to_file("lime_explanation.html")

        # Feature Importance Plot
        exp.as_pyplot_figure()
        plt.savefig("lime_feature_importance_plot.png")
        plt.close()
