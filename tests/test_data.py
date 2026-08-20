import unittest
import pandas as pd

from src.data import load_anime_data, load_rating_data


class TestLoadAnimeData(unittest.TestCase):
    def test_when_called_with_valid_csv_then_returns_dataframe(self):
        result = load_anime_data("data/anime.csv")
        self.assertIsInstance(result, pd.DataFrame)

    def test_when_called_with_valid_csv_then_dataframe_is_not_empty(self):
        result = load_anime_data("data/anime.csv")
        self.assertFalse(result.empty)

    def test_when_called_with_nonexistent_path_then_raises_file_not_found_error(self):
        with self.assertRaises(FileNotFoundError):
            load_anime_data("data/nonexistent.csv")


class TestLoadRatingData(unittest.TestCase):
    def test_when_called_with_valid_csv_then_returns_dataframe(self):
        result = load_rating_data("data/rating.csv")
        self.assertIsInstance(result, pd.DataFrame)

    def test_when_called_with_valid_csv_then_dataframe_is_not_empty(self):
        result = load_rating_data("data/rating.csv")
        self.assertFalse(result.empty)

    def test_when_called_with_nonexistent_path_then_raises_file_not_found_error(self):
        with self.assertRaises(FileNotFoundError):
            load_rating_data("data/nonexistent.csv")


if __name__ == "__main__":
    unittest.main()
