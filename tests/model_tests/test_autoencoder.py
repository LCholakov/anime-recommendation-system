import unittest
import pandas as pd
import numpy as np
import torch

from src.models.autoencoder import (
    AnimeAutoencoder,
    build_user_item_matrix,
    train_autoencoder,
    recommend_autoencoder,
)


class TestAnimeAutoencoder(unittest.TestCase):
    def setUp(self):
        self.input_dim = 20

    def test_when_instantiated_then_returns_module(self):
        model = AnimeAutoencoder(self.input_dim)
        self.assertIsInstance(model, torch.nn.Module)

    def test_when_forward_called_then_output_shape_matches_input(self):
        model = AnimeAutoencoder(self.input_dim)
        x = torch.rand(4, self.input_dim)
        out = model(x)
        self.assertEqual(out.shape, x.shape)

    def test_when_forward_called_then_output_values_are_between_0_and_1(self):
        model = AnimeAutoencoder(self.input_dim)
        x = torch.rand(4, self.input_dim)
        out = model(x)
        self.assertTrue((out >= 0).all() and (out <= 1).all())


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

    def test_when_called_then_missing_ratings_are_zero(self):
        result = build_user_item_matrix(self.train_df)
        self.assertEqual(result.loc[3, 10], 0)

    def test_when_called_then_values_are_normalised_within_range(self):
        # with centering (default): values in [-0.5, 0.5]
        result_centered = build_user_item_matrix(self.train_df, center=True)
        self.assertTrue((result_centered.values >= -0.5).all() and (result_centered.values <= 0.5).all())
        # without centering: values in [0, 1]
        result_raw = build_user_item_matrix(self.train_df, center=False)
        self.assertTrue((result_raw.values >= 0).all() and (result_raw.values <= 1).all())


class TestTrainAutoencoder(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.DataFrame({
            "user_id":  [1, 1, 2, 2, 3, 3],
            "anime_id": [10, 20, 10, 30, 20, 30],
            "rating":   [8,  6,  9,  7,  5,  8],
        })
        self.matrix = build_user_item_matrix(self.train_df)

    def test_when_called_then_returns_trained_model(self):
        model = train_autoencoder(self.matrix, epochs=2, batch_size=2)
        self.assertIsInstance(model, torch.nn.Module)


class TestRecommendAutoencoder(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.DataFrame({
            "user_id":  [1, 1, 2, 2, 3, 3],
            "anime_id": [10, 20, 10, 30, 20, 30],
            "rating":   [8,  6,  9,  7,  5,  8],
        })
        self.matrix = build_user_item_matrix(self.train_df)
        self.model  = train_autoencoder(self.matrix, epochs=2, batch_size=2)

    def test_when_called_then_returns_dataframe(self):
        result = recommend_autoencoder(1, self.model, self.matrix, self.train_df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_when_called_then_already_rated_anime_are_excluded(self):
        result = recommend_autoencoder(1, self.model, self.matrix, self.train_df)
        rated_ids = set(self.train_df[self.train_df["user_id"] == 1]["anime_id"])
        self.assertTrue(set(result["anime_id"]).isdisjoint(rated_ids))

    def test_when_called_with_n_then_returns_at_most_n_results(self):
        result = recommend_autoencoder(1, self.model, self.matrix, self.train_df, n=1)
        self.assertLessEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
