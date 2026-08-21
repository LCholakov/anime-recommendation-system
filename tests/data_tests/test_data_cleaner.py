import unittest
import pandas as pd

from src.data_work.data_cleaner import clean_rating_data, clean_anime_data


class TestCleanRatingData(unittest.TestCase):
    def setUp(self):
        self.valid_anime_ids = pd.Series([10, 20, 30])
        self.df = pd.DataFrame({
            "user_id":  [1,    2,    2,    3,    4,    None, 5   ],
            "anime_id": [10,   20,   20,   30,   99,   10,   10  ],
            "rating":   [5,    8,    8,    11,   7,    6,    -1  ]
        })

    def test_when_called_then_returns_dataframe(self):
        result = clean_rating_data(self.df, self.valid_anime_ids)
        self.assertIsInstance(result, pd.DataFrame)

    def test_when_called_then_no_missing_user_ids(self):
        result = clean_rating_data(self.df, self.valid_anime_ids)
        self.assertFalse(result["user_id"].isna().any())

    def test_when_called_then_no_missing_anime_ids(self):
        result = clean_rating_data(self.df, self.valid_anime_ids)
        self.assertFalse(result["anime_id"].isna().any())

    def test_when_called_then_no_duplicate_entries(self):
        result = clean_rating_data(self.df, self.valid_anime_ids)
        self.assertFalse(result.duplicated(subset=["user_id", "anime_id"]).any())

    def test_when_called_then_ratings_are_within_valid_range(self):
        result = clean_rating_data(self.df, self.valid_anime_ids)
        self.assertTrue(result["rating"].between(1, 10).all())

    def test_when_called_then_no_ratings_referencing_unknown_anime(self):
        result = clean_rating_data(self.df, self.valid_anime_ids)
        self.assertTrue(result["anime_id"].isin(self.valid_anime_ids).all())


class TestCleanAnimeData(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "anime_id": [1,      2,      2,      None,   3    ],
            "name":     ["A",    "B",    "B",    "C",    "D"  ],
            "genre":    ["X",    "Y",    "Y",    "Z",    None ],
            "type":     ["TV",   "Movie","Movie","OVA",  "TV" ],
            "episodes": [12,     1,      1,      6,      24   ],
            "rating":   [7.5,    8.0,    8.0,    6.5,    9.0  ],
            "members":  [1000,   2000,   2000,   500,    3000 ]
        })

    def test_when_called_then_returns_dataframe(self):
        result = clean_anime_data(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_when_called_then_no_duplicate_anime_ids(self):
        result = clean_anime_data(self.df)
        self.assertFalse(result["anime_id"].duplicated().any())

    def test_when_called_then_no_missing_anime_ids(self):
        result = clean_anime_data(self.df)
        self.assertFalse(result["anime_id"].isna().any())

    def test_when_called_then_no_rows_with_missing_fields(self):
        result = clean_anime_data(self.df)
        self.assertFalse(result.isna().any().any())


if __name__ == "__main__":
    unittest.main()
