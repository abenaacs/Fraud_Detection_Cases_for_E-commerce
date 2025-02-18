import pandas as pd


class DataCleaner:
    @staticmethod
    def handle_missing_values(df):
        # Drop rows with missing values
        return df.dropna()

    @staticmethod
    def remove_duplicates(df):
        return df.drop_duplicates()

    @staticmethod
    def correct_data_types(df):
        # Correct data types for specific columns
        if "signup_time" in df.columns:
            df["signup_time"] = pd.to_datetime(df["signup_time"])
        if "purchase_time" in df.columns:
            df["purchase_time"] = pd.to_datetime(df["purchase_time"])
        if "ip_address" in df.columns:
            df["ip_address"] = df["ip_address"].astype(str)
        return df

    @staticmethod
    def impute_missing_values(df, strategy="mean"):
        """
        Imputes missing values using the specified strategy.
        :param df: Input DataFrame
        :param strategy: Imputation strategy ('mean', 'median', 'most_frequent')
        :return: DataFrame with missing values imputed
        """
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(strategy=strategy)
        numeric_columns = df.select_dtypes(include=["number"]).columns
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        return df
