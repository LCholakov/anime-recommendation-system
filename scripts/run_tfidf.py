import sys
import os
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_work.data_loader import load_anime_data
from src.models.tfidf import build_tfidf_matrix, recommend_tfidf
from src.models.evaluator import evaluate, log_run

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MIN_RATING_THRESHOLD = 1      # only use ratings >= this as recommendation seeds
SUBLINEAR_TF         = False  # set True to use 1+log(tf) weighting in TF-IDF

# --- load data ---
anime_df = load_anime_data("data/anime_clean.csv")
train    = pd.read_csv("data/train.csv")
test     = pd.read_csv("data/test.csv")
print(f"Train: {len(train)} rows | Test: {len(test)} rows")

# --- build TF-IDF matrix ---
print(f"Building TF-IDF matrix (sublinear_tf={SUBLINEAR_TF})...")
tfidf_matrix = build_tfidf_matrix(anime_df, sublinear_tf=SUBLINEAR_TF)
print(f"  Matrix shape: {tfidf_matrix.shape} ({tfidf_matrix.shape[1]} terms)")

# --- shared eval sample (same 1000 users across all models) ---
sample_users = pd.read_csv("data/eval_users.csv")["user_id"].tolist()
test_sample  = test[test["user_id"].isin(sample_users)]

print(f"Evaluating on {len(sample_users)} sampled users (n=10)...")
metrics = evaluate(
    lambda user_id, n: recommend_tfidf(user_id, train, tfidf_matrix, anime_df, n=n,
                                       min_rating_threshold=MIN_RATING_THRESHOLD),
    test_sample, train, n=10
)
print(f"  Hit Rate : {metrics['hit_rate']}")
print(f"  Precision: {metrics['precision']}")
print(f"  Recall   : {metrics['recall']}")

# --- save model artifact ---
with open(os.path.join(MODEL_DIR, "tfidf_matrix.pkl"), "wb") as f:
    pickle.dump(tfidf_matrix, f)
print("✅ Saved model/tfidf_matrix.pkl")

# --- record run in tracker ---
# Edit COMMENT before each run to describe what changed.
COMMENT = f"Content-based. TF-IDF genre vectors, L2-normalised. sublinear_tf={SUBLINEAR_TF}, min_rating_threshold={MIN_RATING_THRESHOLD}."
log_run(
    "report/model_performance_tracker.xlsx",
    "TF-IDF + Cosine Similarity",
    metrics,
    {"split": "leave-one-out", "min_ratings": 5, "n_recommendations": 10,
     "min_rating_threshold": MIN_RATING_THRESHOLD, "sublinear_tf": SUBLINEAR_TF},
    COMMENT,
)
print("✅ Results written to report/model_performance_tracker.xlsx")
