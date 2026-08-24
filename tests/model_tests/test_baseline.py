import unittest
import pandas as pd

from src.models.baseline import recommend_popular_anime


class TestRecommendPopularAnime(unittest.TestCase):
    def setUp(self):
        self.anime_df = pd.DataFrame({
            "anime_id": [1,    2,    3,    4,    5   ],
            "name":     ["A",  "B",  "C",  "D",  "E" ],
            "members":  [5000, 3000, 8000, 1000, 6000],
        })
        self.rating_df = pd.DataFrame({
            "user_id":  [1, 1, 1],
            "anime_id": [3, 5, 1],
            "rating":   [9, 8, 7],
        })

    def test_when_recommendation_is_requested_then_most_popular_anime_is_returned_first(self):
        result = recommend_popular_anime(self.anime_df)
        self.assertEqual(result.iloc[0]["anime_id"], 3)

    def test_when_user_is_provided_then_already_rated_anime_are_excluded(self):
        result = recommend_popular_anime(self.anime_df, self.rating_df, user_id=1)
        rated_ids = set(self.rating_df[self.rating_df["user_id"] == 1]["anime_id"])
        self.assertTrue(set(result["anime_id"]).isdisjoint(rated_ids))

    def test_when_n_is_5_then_five_anime_are_returned(self):
        result = recommend_popular_anime(self.anime_df, n=5)
        self.assertEqual(len(result), 5)


if __name__ == "__main__":
    unittest.main()
