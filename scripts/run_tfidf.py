import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_work.data_loader import load_anime_data
from src.models.tfidf import build_tfidf_matrix, recommend_tfidf
from src.models.evaluator import evaluate, append_to_tracker

# --- load data ---
anime_df = load_anime_data("data/anime_clean.csv")
train    = pd.read_csv("data/train.csv")
test     = pd.read_csv("data/test.csv")
print(f"Train: {len(train)} rows | Test: {len(test)} rows")

# --- build TF-IDF matrix ---
print("Building TF-IDF matrix...")
tfidf_matrix = build_tfidf_matrix(anime_df)
print(f"  Matrix shape: {tfidf_matrix.shape} ({tfidf_matrix.shape[1]} terms)")

# --- evaluate on a sample for speed ---
EVAL_USERS   = 1000
test_users   = test["user_id"].unique()
sample_users = pd.Series(test_users).sample(min(EVAL_USERS, len(test_users)), random_state=42).tolist()
test_sample  = test[test["user_id"].isin(sample_users)]

print(f"Evaluating on {len(sample_users)} sampled users (n=10)...")
metrics = evaluate(
    lambda user_id, n: recommend_tfidf(user_id, train, tfidf_matrix, anime_df, n=n),
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
    "TF-IDF + Cosine Similarity",
    5, 0.3, 10, "N/A",
    metrics["hit_rate"], metrics["precision"], metrics["recall"],
    "Content-based. TF-IDF genre vectors, cosine similarity weighted by user rating. Rare genres weighted higher than common ones."
]
append_to_tracker("report/model_performance_tracker.xlsx", row_data, HEADERS)
print("✅ Results written to report/model_performance_tracker.xlsx")
