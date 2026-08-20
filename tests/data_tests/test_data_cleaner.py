import unittest
import os
import pandas as pd

from src.data_work.data_cleaner import clean_rating_data


class TestCleanRatingData(unittest.TestCase):
    def setUp(self):
        self.input_path = "data/rating.csv"
        self.output_path = "data/rating_clean.csv"

    def tearDown(self):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    def test_when_called_with_rating_data_then_returns_dataframe(self):
        result = clean_rating_data(self.input_path, self.output_path)
        self.assertIsInstance(result, pd.DataFrame)

    def test_when_called_with_rating_data_then_negative_ratings_are_removed(self):
        result = clean_rating_data(self.input_path, self.output_path)
        self.assertFalse((result["rating"] == -1).any())

    def test_when_called_with_rating_data_then_clean_csv_is_created(self):
        clean_rating_data(self.input_path, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))


if __name__ == "__main__":
    unittest.main()
