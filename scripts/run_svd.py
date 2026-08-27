import sys
import os
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_work.data_loader import load_anime_data
from src.models.svd import build_user_item_matrix, train_svd, recommend_svd
from src.models.evaluator import evaluate, log_run

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# --- load data ---
anime_df = load_anime_data("data/anime_clean.csv")
train    = pd.read_csv("data/train.csv")
test     = pd.read_csv("data/test.csv")
print(f"Train: {len(train)} rows | Test: {len(test)} rows")

# --- build user-item matrix & train SVD ---
import time
N_COMPONENTS = 50
print(f"Building user-item matrix and training SVD (n_components={N_COMPONENTS})...")
matrix = build_user_item_matrix(train)
_t0 = time.time()
reconstructed, Vt, anime_cols = train_svd(matrix, n_components=N_COMPONENTS)
TRAIN_SECS = round(time.time() - _t0, 1)
print(f"  Matrix shape: {matrix.shape}")
print(f"  Train time: {TRAIN_SECS}s")

# --- shared eval sample (same 1000 users across all models) ---
# keep only those present in the reconstructed matrix (should be all of them)
sample_users = pd.read_csv("data/eval_users.csv")["user_id"].tolist()
sample_users = [u for u in sample_users if u in reconstructed.index]
test_sample  = test[test["user_id"].isin(sample_users)]

print(f"Evaluating on {len(sample_users)} users (n=10)...")
metrics = evaluate(
    lambda user_id, n: recommend_svd(user_id, reconstructed, train, n=n),
    test_sample, train, n=10
)
print(f"  Hit Rate : {metrics['hit_rate']}")
print(f"  Precision: {metrics['precision']}")
print(f"  Recall   : {metrics['recall']}")

# --- save model artifact ---
with open(os.path.join(MODEL_DIR, "svd_reconstructed.pkl"), "wb") as f:
    pickle.dump((reconstructed, Vt, anime_cols), f)
print("✅ Saved model/svd_reconstructed.pkl")

# --- record run in tracker ---
# Edit COMMENT before each run to describe what changed.
COMMENT = f"Collaborative filtering. TruncatedSVD n_components={N_COMPONENTS}. Factorises user-item rating matrix."
log_run(
    "report/model_performance_tracker.xlsx",
    "SVD Collaborative Filtering",
    metrics,
    {"split": "leave-one-out", "min_ratings": 5, "n_recommendations": 10,
     "n_components": N_COMPONENTS, "train_secs": TRAIN_SECS},
    COMMENT,
)
print("✅ Results written to report/model_performance_tracker.xlsx")
