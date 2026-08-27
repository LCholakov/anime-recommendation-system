import sys
import os
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_work.data_loader import load_anime_data, load_rating_data
from src.models.baseline import split_train_test
from src.models.bow import build_bow_matrix, recommend_bow
from src.models.evaluator import evaluate, log_run

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

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

# --- shared eval sample (same 1000 users across all models) ---
sample_users = pd.read_csv("data/eval_users.csv")["user_id"].tolist()
test_sample  = test[test["user_id"].isin(sample_users)]

print(f"Evaluating on {len(sample_users)} sampled users (n=10)...")
metrics = evaluate(
    lambda user_id, n: recommend_bow(user_id, train, bow_matrix, anime_df, n=n),
    test_sample, train, n=10
)
print(f"  Hit Rate : {metrics['hit_rate']}")
print(f"  Precision: {metrics['precision']}")
print(f"  Recall   : {metrics['recall']}")

# --- save model artifact ---
with open(os.path.join(MODEL_DIR, "bow_matrix.pkl"), "wb") as f:
    pickle.dump(bow_matrix, f)
print("✅ Saved model/bow_matrix.pkl")

# --- record run in tracker ---
# Edit COMMENT before each run to describe what changed.
COMMENT = "Content-based. Genre BoW, L2-normalised, dot-product cosine. Weighted by user rating."
log_run(
    "report/model_performance_tracker.xlsx",
    "BoW + Cosine Similarity",
    metrics,
    {"split": "leave-one-out", "min_ratings": 5, "n_recommendations": 10},
    COMMENT,
)
print("✅ Results written to report/model_performance_tracker.xlsx")
