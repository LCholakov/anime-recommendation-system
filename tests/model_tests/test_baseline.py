import unittest
import pandas as pd

from src.models.baseline import split_train_test, compute_bayesian_scores, recommend_popular_anime, evaluate_model


class TestSplitTrainTest(unittest.TestCase):
    def setUp(self):
        self.ratings_df = pd.DataFrame({
            "user_id":  [1, 1, 1, 1, 1, 1,  2, 2, 2, 2, 2, 2,  3, 3, 3],
            "anime_id": [1, 2, 3, 4, 5, 6,  1, 2, 3, 4, 5, 6,  1, 2, 3],
            "rating":   [5, 6, 7, 8, 9, 8,  4, 5, 6, 7, 8, 9,  5, 6, 7],
        })

    def test_when_called_then_returns_train_and_test(self):
        train, test = split_train_test(self.ratings_df, min_ratings=5)
        self.assertIsInstance(train, pd.DataFrame)
        self.assertIsInstance(test, pd.DataFrame)

    def test_when_called_then_users_with_fewer_than_min_ratings_are_excluded(self):
        train, test = split_train_test(self.ratings_df, min_ratings=5)
        all_users = set(train["user_id"]) | set(test["user_id"])
        for user in all_users:
            total = len(self.ratings_df[self.ratings_df["user_id"] == user])
            self.assertGreaterEqual(total, 5)

    def test_when_called_then_no_user_appears_in_both_train_and_test_for_same_anime(self):
        train, test = split_train_test(self.ratings_df, min_ratings=5)
        merged = train.merge(test, on=["user_id", "anime_id"])
        self.assertTrue(merged.empty)


class TestComputeBayesianScores(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.DataFrame({
            "user_id":  [1, 1, 2, 2, 3],
            "anime_id": [1, 2, 1, 3, 2],
            "rating":   [8, 6, 9, 7, 5],
        })

    def test_when_called_then_returns_dataframe(self):
        result = compute_bayesian_scores(self.train_df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_when_called_then_result_has_bayesian_score_column(self):
        result = compute_bayesian_scores(self.train_df)
        self.assertIn("bayesian_score", result.columns)

    def test_when_called_then_scores_are_between_1_and_10(self):
        result = compute_bayesian_scores(self.train_df)
        self.assertTrue(result["bayesian_score"].between(1, 10).all())


class TestRecommendPopularAnime(unittest.TestCase):
    def setUp(self):
        self.scores_df = pd.DataFrame({
            "anime_id":       [1,    2,    3,    4,    5   ],
            "bayesian_score": [7.5,  6.0,  9.0,  5.5,  8.0],
        })
        self.rating_df = pd.DataFrame({
            "user_id":  [1, 1, 1],
            "anime_id": [3, 5, 1],
            "rating":   [9, 8, 7],
        })

    def test_when_recommendation_is_requested_then_highest_scored_anime_is_returned_first(self):
        result = recommend_popular_anime(self.scores_df)
        self.assertEqual(result.iloc[0]["anime_id"], 3)

    def test_when_user_is_provided_then_already_rated_anime_are_excluded(self):
        result = recommend_popular_anime(self.scores_df, self.rating_df, user_id=1)
        rated_ids = set(self.rating_df[self.rating_df["user_id"] == 1]["anime_id"])
        self.assertTrue(set(result["anime_id"]).isdisjoint(rated_ids))

    def test_when_n_is_3_then_three_anime_are_returned(self):
        result = recommend_popular_anime(self.scores_df, n=3)
        self.assertEqual(len(result), 3)


class TestEvaluateModel(unittest.TestCase):
    def setUp(self):
        # scores_df: anime 3 is top, then 5, then 1
        self.scores_df = pd.DataFrame({
            "anime_id":       [1,   2,   3,   4,   5  ],
            "bayesian_score": [7.5, 6.0, 9.0, 5.5, 8.0],
        })
        # test set: user 1 liked anime 3 and 5 (in top 10 recommendations)
        # user 2 liked anime 4 (not in top 3 recommendations)
        self.test_df = pd.DataFrame({
            "user_id":  [1, 1, 2],
            "anime_id": [3, 5, 4],
            "rating":   [9, 8, 7],
        })
        self.train_df = pd.DataFrame({
            "user_id":  [1, 1, 2, 2],
            "anime_id": [1, 2, 1, 2],
            "rating":   [5, 6, 7, 8],
        })

    def test_when_called_then_returns_dict(self):
        result = evaluate_model(self.scores_df, self.test_df, self.train_df)
        self.assertIsInstance(result, dict)

    def test_when_called_then_result_has_precision_recall_hit_rate(self):
        result = evaluate_model(self.scores_df, self.test_df, self.train_df)
        for key in ["precision", "recall", "hit_rate"]:
            self.assertIn(key, result)

    def test_when_all_test_items_are_in_recommendations_then_hit_rate_is_1(self):
        # user 1 has both test items (3, 5) in top 10 → hit
        result = evaluate_model(self.scores_df, self.test_df[self.test_df["user_id"] == 1], self.train_df, n=10)
        self.assertEqual(result["hit_rate"], 1.0)

    def test_when_no_test_items_are_in_recommendations_then_hit_rate_is_0(self):
        # user 2's test item is anime 4 (score 5.5), but n=2 only returns anime 3 and 5
        result = evaluate_model(self.scores_df, self.test_df[self.test_df["user_id"] == 2], self.train_df, n=2)
        self.assertEqual(result["hit_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
