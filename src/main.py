from data_loader import DataLoader
from data_cleaner import DataCleaner
from feature_engineer import FeatureEngineer
from model_builder import ModelBuilder
from mlflow_utils import setup_mlflow


def main():
    # Step 1: Initialize DataLoader
    loader = DataLoader(
        fraud_data_path="data/Fraud_Data.csv",
        ip_country_path="data/IpAddress_to_Country.csv",
        creditcard_path="data/creditcard.csv",
    )

    # Step 2: Load datasets
    fraud_df = loader.load_fraud_data()
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
    fraud_df = engineer.encode_categorical_features(
        fraud_df, ["source", "browser", "sex"]
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


if __name__ == "__main__":
    main()
