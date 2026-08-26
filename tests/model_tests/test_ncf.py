import unittest
import pandas as pd
import numpy as np
import torch

from src.models.ncf import NCF, encode_ids, train_ncf, recommend_ncf


class TestEncodeIds(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.DataFrame({
            "user_id":  [1, 1, 2, 3],
            "anime_id": [10, 20, 10, 30],
            "rating":   [8,  6,  9,  7 ],
        })

    def test_when_called_then_returns_two_dicts(self):
        user_map, anime_map = encode_ids(self.train_df)
        self.assertIsInstance(user_map, dict)
        self.assertIsInstance(anime_map, dict)

    def test_when_called_then_all_users_are_mapped(self):
        user_map, _ = encode_ids(self.train_df)
        for uid in self.train_df["user_id"].unique():
            self.assertIn(uid, user_map)

    def test_when_called_then_all_anime_are_mapped(self):
        _, anime_map = encode_ids(self.train_df)
        for aid in self.train_df["anime_id"].unique():
            self.assertIn(aid, anime_map)


class TestNCF(unittest.TestCase):
    def setUp(self):
        self.n_users = 5
        self.n_anime = 10
        self.model   = NCF(self.n_users, self.n_anime)

    def test_when_instantiated_then_returns_module(self):
        self.assertIsInstance(self.model, torch.nn.Module)

    def test_when_forward_called_then_output_shape_is_correct(self):
        users  = torch.tensor([0, 1, 2])
        anime  = torch.tensor([0, 1, 2])
        out    = self.model(users, anime)
        self.assertEqual(out.shape, (3,))

    def test_when_forward_called_then_output_is_float(self):
        users = torch.tensor([0])
        anime = torch.tensor([0])
        out   = self.model(users, anime)
        self.assertEqual(out.dtype, torch.float32)


class TestTrainNcf(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.DataFrame({
            "user_id":  [1, 1, 2, 2, 3, 3],
            "anime_id": [10, 20, 10, 30, 20, 30],
            "rating":   [8,  6,  9,  7,  5,  8 ],
        })

    def test_when_called_then_returns_model_and_maps(self):
        model, user_map, anime_map = train_ncf(self.train_df, epochs=2, batch_size=3)
        self.assertIsInstance(model, torch.nn.Module)
        self.assertIsInstance(user_map, dict)
        self.assertIsInstance(anime_map, dict)


class TestRecommendNcf(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.DataFrame({
            "user_id":  [1, 1, 2, 2, 3, 3],
            "anime_id": [10, 20, 10, 30, 20, 30],
            "rating":   [8,  6,  9,  7,  5,  8 ],
        })
        self.model, self.user_map, self.anime_map = train_ncf(
            self.train_df, epochs=2, batch_size=3
        )

    def test_when_called_then_returns_dataframe(self):
        result = recommend_ncf(1, self.model, self.user_map, self.anime_map, self.train_df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_when_called_then_already_rated_anime_are_excluded(self):
        result = recommend_ncf(1, self.model, self.user_map, self.anime_map, self.train_df)
        rated_ids = set(self.train_df[self.train_df["user_id"] == 1]["anime_id"])
        self.assertTrue(set(result["anime_id"]).isdisjoint(rated_ids))

    def test_when_called_with_n_then_returns_at_most_n_results(self):
        result = recommend_ncf(1, self.model, self.user_map, self.anime_map, self.train_df, n=1)
        self.assertLessEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
