import unittest
import pandas as pd
import numpy as np

from src.models.bow import build_bow_matrix, get_similar_anime, recommend_bow


class TestBuildBowMatrix(unittest.TestCase):
    def setUp(self):
        self.anime_df = pd.DataFrame({
            "anime_id": [1,                    2,                  3,               4              ],
            "name":     ["A",                  "B",                "C",             "D"            ],
            "genre":    ["Action, Adventure",  "Action, Comedy",   "Romance, Drama","Adventure, Drama"],
        })

    def test_when_called_then_returns_dataframe(self):
        result = build_bow_matrix(self.anime_df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_when_called_then_index_matches_anime_ids(self):
        result = build_bow_matrix(self.anime_df)
        self.assertListEqual(list(result.index), list(self.anime_df["anime_id"]))

    def test_when_called_then_columns_are_unique_genre_words(self):
        result = build_bow_matrix(self.anime_df)
        self.assertIn("action", result.columns)
        self.assertIn("adventure", result.columns)

    def test_when_called_then_values_are_in_unit_range(self):
        result = build_bow_matrix(self.anime_df)
        self.assertTrue((result.values >= 0.0).all())
        self.assertTrue((result.values <= 1.0).all())


class TestGetSimilarAnime(unittest.TestCase):
    def setUp(self):
        self.anime_df = pd.DataFrame({
            "anime_id": [1,                    2,                  3,               4              ],
            "name":     ["A",                  "B",                "C",             "D"            ],
            "genre":    ["Action, Adventure",  "Action, Comedy",   "Romance, Drama","Adventure, Drama"],
        })
        self.bow_matrix = build_bow_matrix(self.anime_df)

    def test_when_called_then_returns_dataframe(self):
        result = get_similar_anime(1, self.bow_matrix, self.anime_df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_when_called_then_query_anime_is_not_in_results(self):
        result = get_similar_anime(1, self.bow_matrix, self.anime_df)
        self.assertNotIn(1, result["anime_id"].values)

    def test_when_called_then_results_are_sorted_by_similarity_descending(self):
        result = get_similar_anime(1, self.bow_matrix, self.anime_df)
        scores = result["similarity"].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_when_called_with_n_then_returns_n_results(self):
        result = get_similar_anime(1, self.bow_matrix, self.anime_df, n=2)
        self.assertEqual(len(result), 2)


class TestRecommendBow(unittest.TestCase):
    def setUp(self):
        self.anime_df = pd.DataFrame({
            "anime_id": [1,                    2,                  3,               4              ],
            "name":     ["A",                  "B",                "C",             "D"            ],
            "genre":    ["Action, Adventure",  "Action, Comedy",   "Romance, Drama","Adventure, Drama"],
        })
        self.bow_matrix = build_bow_matrix(self.anime_df)
        self.train_df = pd.DataFrame({
            "user_id":  [1, 1],
            "anime_id": [1, 2],
            "rating":   [9, 8],
        })

    def test_when_called_then_returns_dataframe(self):
        result = recommend_bow(1, self.train_df, self.bow_matrix, self.anime_df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_when_called_then_already_rated_anime_are_excluded(self):
        result = recommend_bow(1, self.train_df, self.bow_matrix, self.anime_df)
        rated_ids = set(self.train_df[self.train_df["user_id"] == 1]["anime_id"])
        self.assertTrue(set(result["anime_id"]).isdisjoint(rated_ids))

    def test_when_called_with_n_then_returns_at_most_n_results(self):
        result = recommend_bow(1, self.train_df, self.bow_matrix, self.anime_df, n=2)
        self.assertLessEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
