import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_work.data_loader import load_anime_data, load_rating_data
from src.models.baseline import split_train_test, compute_bayesian_scores, recommend_popular_anime
from src.models.evaluator import evaluate, append_to_tracker

# --- load & split ---
ratings_df = load_rating_data("data/rating_clean.csv")
anime_df   = load_anime_data("data/anime_clean.csv")

print("Splitting train/test (min 5 ratings per user, 70/30)...")
train, test = split_train_test(ratings_df, min_ratings=5, test_ratio=0.3)
print(f"  Train: {len(train)} rows | Test: {len(test)} rows")

# --- compute scores ---
print("Computing Bayesian scores from training data...")
scores = compute_bayesian_scores(train)
scores_named = scores.merge(anime_df[["anime_id", "name", "genre"]], on="anime_id", how="left")
scores_named = scores_named.sort_values("bayesian_score", ascending=False).reset_index(drop=True)

# --- evaluate ---
print("Evaluating (n=10)...")
metrics = evaluate(
    lambda user_id, n: recommend_popular_anime(scores, train, user_id=user_id, n=n),
    test, train, n=10
)
print(f"  Hit Rate : {metrics['hit_rate']}")
print(f"  Precision: {metrics['precision']}")
print(f"  Recall   : {metrics['recall']}")

# --- save data ---
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
scores_named.to_csv("data/baseline_scores.csv", index=False)
print(f"✅ Saved data/train.csv, data/test.csv, data/baseline_scores.csv")

print("\nTop 10 by Bayesian score:")
print(scores_named[["name", "rating_count", "avg_rating", "bayesian_score"]].head(10).to_string(index=False))

# --- write to model performance tracker ---
HEADERS = [
    "Model", "min_ratings", "test_ratio", "n_recommendations",
    "m (percentile)", "Hit Rate @10", "Precision @10", "Recall @10", "Comments"
]
row_data = [
    "Baseline (Popularity)",
    5, 0.3, 10, "80th pct",
    metrics["hit_rate"], metrics["precision"], metrics["recall"],
    "Greedy baseline. Recommends same top-10 popular anime to all users. No personalisation."
]
append_to_tracker("report/model_performance_tracker.xlsx", row_data, HEADERS)
print("✅ Results written to report/model_performance_tracker.xlsx")
