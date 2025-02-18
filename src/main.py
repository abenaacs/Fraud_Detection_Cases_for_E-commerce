from data_loader import DataLoader
from data_cleaner import DataCleaner
from feature_engineer import FeatureEngineer
from model_builder import ModelBuilder
from mlflow_utils import setup_mlflow
from explainability import Explainability


def main():
    # Step 1: Initialize DataLoader
    loader = DataLoader(
        fraud_data_path="data/Fraud_Data.csv",
        ip_country_path="data/IpAddress_to_Country.csv",
        creditcard_path="data/creditcard.csv",
    )

    # Step 2: Load datasets
    fraud_df = loader.load_fraud_data()
    # Compute correlation with the target variable
    correlation = fraud_df.corr(numeric_only=True)["class"].sort_values(ascending=False)
    print(correlation)
    ip_country_df = loader.load_ip_country_data()
    creditcard_df = loader.load_creditcard_data()

    # Step 3: Clean Fraud Data
    cleaner = DataCleaner()
    fraud_df = cleaner.handle_missing_values(fraud_df)
    fraud_df = cleaner.remove_duplicates(fraud_df)
    fraud_df = cleaner.correct_data_types(fraud_df)

    # Step 4: Feature Engineering for Fraud Data
    engineer = FeatureEngineer()
    fraud_df = engineer.add_time_features(fraud_df)
    fraud_df = engineer.merge_with_geolocation(fraud_df, ip_country_df)
    # Drop unnecessary columns (timestamps and others not needed for modeling)
    fraud_df = FeatureEngineer.drop_unnecessary_columns(
        fraud_df,
        columns_to_drop=[
            "signup_time",
            "purchase_time",
            "device_id",
            "ip_address",
            "lower_bound_ip_address",
            "upper_bound_ip_address",
        ],
    )

    fraud_df = engineer.encode_categorical_features(
        fraud_df, ["source", "browser", "sex", "country"]
    )

    # Step 5: Save cleaned and processed data
    fraud_df.to_csv("data/processed_fraud_data.csv", index=False)
    creditcard_df.to_csv("data/processed_creditcard_data.csv", index=False)

    # Step 6: Set up MLflow for experiment tracking
    setup_mlflow(experiment_name="Fraud_Detection_Experiment")

    # Step 7: Build and evaluate models for Fraud Data
    fraud_model_builder = ModelBuilder(
        data_path="data/processed_fraud_data.csv", target_column="class"
    )
    fraud_model_builder.split_data(test_size=0.2, random_state=42)
    fraud_results = fraud_model_builder.train_and_evaluate()

    # Step 8: Build and evaluate models for Credit Card Data
    creditcard_model_builder = ModelBuilder(
        data_path="data/processed_creditcard_data.csv", target_column="Class"
    )
    creditcard_model_builder.split_data(test_size=0.2, random_state=42)
    creditcard_results = creditcard_model_builder.train_and_evaluate()

    # Step 9: Print results
    print("Fraud Data Results:")
    for model, metrics in fraud_results.items():
        print(f"{model}: {metrics}")
    print("\nCredit Card Data Results:")
    for model, metrics in creditcard_results.items():
        print(f"{model}: {metrics}")

    # Step 10: Explain the best-performing model for Fraud Data
    best_fraud_model_name = max(
        fraud_results, key=lambda x: fraud_results[x]["f1_score"]
    )
    best_fraud_model = fraud_model_builder.models[best_fraud_model_name]
    print(f"\nBest Fraud Model: {best_fraud_model_name}")

    fraud_explainer = Explainability(
        model=best_fraud_model,
        X_train=fraud_model_builder.X_train,
        X_test=fraud_model_builder.X_test,
        feature_names=fraud_model_builder.X_train.columns,
    )
    fraud_explainer.explain_with_shap()
    fraud_explainer.explain_with_lime()
    print("SHAP and LIME explanations generated for Fraud Data.")

    # Step 11: Explain the best-performing model for Credit Card Data
    best_creditcard_model_name = max(
        creditcard_results, key=lambda x: creditcard_results[x]["f1_score"]
    )
    best_creditcard_model = creditcard_model_builder.models[best_creditcard_model_name]
    print(f"\nBest Credit Card Model: {best_creditcard_model_name}")

    creditcard_explainer = Explainability(
        model=best_creditcard_model,
        X_train=creditcard_model_builder.X_train,
        X_test=creditcard_model_builder.X_test,
        feature_names=creditcard_model_builder.X_train.columns,
    )
    creditcard_explainer.explain_with_shap()
    creditcard_explainer.explain_with_lime()
    print("SHAP and LIME explanations generated for Credit Card Data.")


if __name__ == "__main__":
    main()
