import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_work.data_loader import load_anime_data
from src.models.svd import build_user_item_matrix, train_svd, recommend_svd
from src.models.evaluator import evaluate, append_to_tracker

# --- load data ---
anime_df = load_anime_data("data/anime_clean.csv")
train    = pd.read_csv("data/train.csv")
test     = pd.read_csv("data/test.csv")
print(f"Train: {len(train)} rows | Test: {len(test)} rows")

# --- build user-item matrix & train SVD ---
N_COMPONENTS = 50
print(f"Building user-item matrix and training SVD (n_components={N_COMPONENTS})...")
matrix       = build_user_item_matrix(train)
reconstructed = train_svd(matrix, n_components=N_COMPONENTS)
print(f"  Matrix shape: {matrix.shape}")

# --- evaluate on a sample for speed ---
EVAL_USERS   = 1000
test_users   = test["user_id"].unique()
sample_users = pd.Series(test_users).sample(min(EVAL_USERS, len(test_users)), random_state=42).tolist()
# keep only users present in the reconstructed matrix
sample_users = [u for u in sample_users if u in reconstructed.index]
test_sample  = test[test["user_id"].isin(sample_users)]

print(f"Evaluating on {len(sample_users)} sampled users (n=10)...")
metrics = evaluate(
    lambda user_id, n: recommend_svd(user_id, reconstructed, train, n=n),
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
    "SVD Collaborative Filtering",
    5, 0.3, 10, "N/A",
    metrics["hit_rate"], metrics["precision"], metrics["recall"],
    f"Collaborative filtering. TruncatedSVD n_components={N_COMPONENTS}. Factorises user-item rating matrix."
]
append_to_tracker("report/model_performance_tracker.xlsx", row_data, HEADERS)
print("✅ Results written to report/model_performance_tracker.xlsx")
