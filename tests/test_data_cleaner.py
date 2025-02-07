import unittest
import pandas as pd
from src.data_cleaner import DataCleaner

class TestDataCleaner(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'signup_time': ['2023-01-01 12:00:00', None],
            'purchase_time': ['2023-01-01 12:05:00', '2023-01-01 12:10:00'],
            'ip_address': ['192.168.1.1', '192.168.1.2'],
            'duplicate_col': [1, 1]
        })

    def test_handle_missing_values(self):
        cleaner = DataCleaner()
        cleaned_df = cleaner.handle_missing_values(self.df)
        self.assertEqual(len(cleaned_df), 1)  # Missing row dropped

    def test_remove_duplicates(self):
        cleaner = DataCleaner()
        cleaned_df = cleaner.remove_duplicates(self.df)
        self.assertEqual(len(cleaned_df), 1)  # Duplicate row removed

    def test_correct_data_types(self):
        cleaner = DataCleaner()
        cleaned_df = cleaner.correct_data_types(self.df)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned_df['signup_time']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned_df['purchase_time']))

if __name__ == "__main__":
    unittest.main()