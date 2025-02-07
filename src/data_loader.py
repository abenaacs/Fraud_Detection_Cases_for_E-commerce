import pandas as pd

class DataLoader:
    def __init__(self, fraud_data_path, ip_country_path, creditcard_path):
        self.fraud_data_path = fraud_data_path
        self.ip_country_path = ip_country_path
        self.creditcard_path = creditcard_path

    def load_fraud_data(self):
        return pd.read_csv(self.fraud_data_path)

    def load_ip_country_data(self):
        return pd.read_csv(self.ip_country_path)

    def load_creditcard_data(self):
        return pd.read_csv(self.creditcard_path)