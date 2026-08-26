import unittest
import pandas as pd
import numpy as np

from src.models.svd import build_user_item_matrix, train_svd, recommend_svd


class TestBuildUserItemMatrix(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.DataFrame({
            "user_id":  [1, 1, 2, 2, 3],
            "anime_id": [10, 20, 10, 30, 20],
            "rating":   [8,  6,  9,  7,  5 ],
        })

    def test_when_called_then_returns_dataframe(self):
        result = build_user_item_matrix(self.train_df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_when_called_then_rows_are_users(self):
        result = build_user_item_matrix(self.train_df)
        self.assertListEqual(sorted(result.index.tolist()), [1, 2, 3])

    def test_when_called_then_columns_are_anime(self):
        result = build_user_item_matrix(self.train_df)
        self.assertListEqual(sorted(result.columns.tolist()), [10, 20, 30])

    def test_when_called_then_missing_ratings_are_zero(self):
        result = build_user_item_matrix(self.train_df)
        self.assertEqual(result.loc[3, 10], 0)


class TestTrainSvd(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.DataFrame({
            "user_id":  [1, 1, 2, 2, 3],
            "anime_id": [10, 20, 10, 30, 20],
            "rating":   [8,  6,  9,  7,  5 ],
        })
        self.matrix = build_user_item_matrix(self.train_df)

    def test_when_called_then_returns_reconstructed_matrix(self):
        result = train_svd(self.matrix, n_components=2)
        self.assertIsInstance(result, pd.DataFrame)

    def test_when_called_then_shape_matches_input(self):
        result = train_svd(self.matrix, n_components=2)
        self.assertEqual(result.shape, self.matrix.shape)

    def test_when_called_then_index_and_columns_match_input(self):
        result = train_svd(self.matrix, n_components=2)
        self.assertListEqual(list(result.index), list(self.matrix.index))
        self.assertListEqual(list(result.columns), list(self.matrix.columns))


class TestRecommendSvd(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.DataFrame({
            "user_id":  [1, 1, 2, 2, 3],
            "anime_id": [10, 20, 10, 30, 20],
            "rating":   [8,  6,  9,  7,  5 ],
        })
        matrix = build_user_item_matrix(self.train_df)
        self.reconstructed = train_svd(matrix, n_components=2)

    def test_when_called_then_returns_dataframe(self):
        result = recommend_svd(1, self.reconstructed, self.train_df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_when_called_then_already_rated_anime_are_excluded(self):
        result = recommend_svd(1, self.reconstructed, self.train_df)
        rated_ids = set(self.train_df[self.train_df["user_id"] == 1]["anime_id"])
        self.assertTrue(set(result["anime_id"]).isdisjoint(rated_ids))

    def test_when_called_with_n_then_returns_at_most_n_results(self):
        result = recommend_svd(1, self.reconstructed, self.train_df, n=1)
        self.assertLessEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
