import pandas as pd


class FeatureEngineer:
    @staticmethod
    def add_time_features(df):
        if "purchase_time" in df.columns:
            df["hour_of_day"] = df["purchase_time"].dt.hour
            df["day_of_week"] = df["purchase_time"].dt.dayofweek
        return df

    @staticmethod
    def merge_with_geolocation(fraud_df, ip_country_df):
        # Convert IP addresses to integers
        fraud_df["ip_address_int"] = fraud_df["ip_address"].apply(
            lambda x: int("".join([f"{int(octet):08b}" for octet in x.split(".")]), 2)
        )

        # Merge datasets based on IP address range
        merged_df = pd.merge(
            fraud_df,
            ip_country_df,
            how="left",
            left_on="ip_address_int",
            right_on="lower_bound_ip_address",
        )
        merged_df["country"] = merged_df["country"].fillna("Unknown")
        return merged_df

    @staticmethod
    def encode_categorical_features(df, categorical_columns):
        # One-hot encoding for categorical features
        return pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    @staticmethod
    def drop_unnecessary_columns(df, columns_to_drop):
        return df.drop(columns=columns_to_drop, errors="ignore")
