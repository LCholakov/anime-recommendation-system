import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_work.data_loader import load_anime_data, load_rating_data
from src.models.baseline import split_train_test
from src.models.bow import build_bow_matrix, recommend_bow
from src.models.evaluator import evaluate, append_to_tracker

# --- load & split ---
ratings_df = load_rating_data("data/rating_clean.csv")
anime_df   = load_anime_data("data/anime_clean.csv")

print("Loading train/test split...")
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")
print(f"  Train: {len(train)} rows | Test: {len(test)} rows")

# --- build BoW matrix ---
print("Building Bag-of-Words matrix...")
bow_matrix = build_bow_matrix(anime_df)
print(f"  Matrix shape: {bow_matrix.shape} ({bow_matrix.shape[1]} genre words)")

# --- evaluate on a sample for speed ---
EVAL_USERS = 1000
test_users  = test["user_id"].unique()
sample_users = pd.Series(test_users).sample(min(EVAL_USERS, len(test_users)), random_state=42).tolist()
test_sample  = test[test["user_id"].isin(sample_users)]

print(f"Evaluating on {len(sample_users)} sampled users (n=10)...")
metrics = evaluate(
    lambda user_id, n: recommend_bow(user_id, train, bow_matrix, anime_df, n=n),
    test_sample, train, n=10
)
print(f"  Hit Rate : {metrics['hit_rate']}")
print(f"  Precision: {metrics['precision']}")
print(f"  Recall   : {metrics['recall']}")

# --- write to model performance tracker ---
HEADERS = [
    "Model", "min_ratings", "test_ratio", "n_recommendations",
    "m (percentile)", "Hit Rate @10", "Precision @10", "Recall @10", "Comments"
]
row_data = [
    "BoW + Cosine Similarity",
    5, 0.3, 10, "N/A",
    metrics["hit_rate"], metrics["precision"], metrics["recall"],
    "Content-based. Genre BoW vectors, cosine similarity weighted by user rating. No collaborative signal."
]
append_to_tracker("report/model_performance_tracker.xlsx", row_data, HEADERS)
print("✅ Results written to report/model_performance_tracker.xlsx")
