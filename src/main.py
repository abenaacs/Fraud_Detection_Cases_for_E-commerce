from data_loader import DataLoader
from data_cleaner import DataCleaner
from feature_engineer import FeatureEngineer

def main():
    # Initialize DataLoader
    loader = DataLoader(
        fraud_data_path="data/Fraud_Data.csv",
        ip_country_path="data/IpAddress_to_Country.csv",
        creditcard_path="data/creditcard.csv"
    )

    # Load datasets
    fraud_df = loader.load_fraud_data()
    ip_country_df = loader.load_ip_country_data()
    creditcard_df = loader.load_creditcard_data()

    # Clean Fraud Data
    cleaner = DataCleaner()
    fraud_df = cleaner.handle_missing_values(fraud_df)
    fraud_df = cleaner.remove_duplicates(fraud_df)
    fraud_df = cleaner.correct_data_types(fraud_df)

    # Feature Engineering
    engineer = FeatureEngineer()
    fraud_df = engineer.add_time_features(fraud_df)
    fraud_df = engineer.merge_with_geolocation(fraud_df, ip_country_df)
    fraud_df = engineer.encode_categorical_features(fraud_df, ['source', 'browser', 'sex'])

    # Save cleaned and processed data
    fraud_df.to_csv("data/processed_fraud_data.csv", index=False)
    creditcard_df.to_csv("data/processed_creditcard_data.csv", index=False)

if __name__ == "__main__":
    main()