import unittest
import pandas as pd
from src.feature_engineer import FeatureEngineer

class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'purchase_time': ['2023-01-01 12:00:00', '2023-01-01 15:30:00'],
            'ip_address': ['192.168.1.1', '192.168.1.2']
        })
        self.ip_country_df = pd.DataFrame({
            'lower_bound_ip_address': [3232235777, 3232235778],
            'upper_bound_ip_address': [3232235777, 3232235778],
            'country': ['US', 'CA']
        })

    def test_add_time_features(self):
        engineer = FeatureEngineer()
        df_with_time = engineer.add_time_features(self.df)
        self.assertIn('hour_of_day', df_with_time.columns)
        self.assertIn('day_of_week', df_with_time.columns)

    def test_merge_with_geolocation(self):
        engineer = FeatureEngineer()
        merged_df = engineer.merge_with_geolocation(self.df, self.ip_country_df)
        self.assertIn('country', merged_df.columns)
        self.assertEqual(merged_df.iloc[0]['country'], 'US')

if __name__ == "__main__":
    unittest.main()