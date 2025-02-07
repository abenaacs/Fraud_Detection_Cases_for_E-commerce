import unittest
from src.data_loader import DataLoader
import pandas as pd

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader(
            fraud_data_path="data/Fraud_Data.csv",
            ip_country_path="data/IpAddress_to_Country.csv",
            creditcard_path="data/creditcard.csv"
        )

    def test_load_fraud_data(self):
        df = self.loader.load_fraud_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('user_id', df.columns)

    def test_load_ip_country_data(self):
        df = self.loader.load_ip_country_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('lower_bound_ip_address', df.columns)

    def test_load_creditcard_data(self):
        df = self.loader.load_creditcard_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('Time', df.columns)

if __name__ == "__main__":
    unittest.main()