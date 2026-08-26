import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from src.data_work.data_loader import load_anime_data
from src.models.autoencoder import build_user_item_matrix, train_autoencoder, recommend_autoencoder
from src.models.evaluator import evaluate, log_run

SEED         = 42
TRAIN_USERS  = 10000   # subsample for tractable matrix size
EVAL_USERS   = 1000
N_COMPONENTS = None    # not applicable for autoencoder
EPOCHS       = 20
BATCH_SIZE   = 128
PATIENCE     = 3

torch.manual_seed(SEED)

# --- load data ---
anime_df = load_anime_data("data/anime_clean.csv")
train    = pd.read_csv("data/train.csv")
test     = pd.read_csv("data/test.csv")
print(f"Train: {len(train)} rows | Test: {len(test)} rows")

# --- subsample training users ---
all_train_users = train["user_id"].unique()
sampled_train_users = pd.Series(all_train_users).sample(
    min(TRAIN_USERS, len(all_train_users)), random_state=SEED
).tolist()
train_sub = train[train["user_id"].isin(sampled_train_users)]
print(f"Using {len(sampled_train_users)} training users for matrix ({len(train_sub)} ratings)")

# --- build user-item matrix ---
print("Building user-item matrix...")
matrix = build_user_item_matrix(train_sub)
print(f"  Matrix shape: {matrix.shape}")

# --- train autoencoder ---
print(f"Training autoencoder (epochs={EPOCHS}, batch={BATCH_SIZE}, patience={PATIENCE})...")
_t0 = time.time()
model = train_autoencoder(matrix, epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE)
TRAIN_SECS = round(time.time() - _t0, 1)
print(f"  Train time: {TRAIN_SECS}s")

# --- shared eval sample (same 1000 users across all models) ---
# AE only saw TRAIN_USERS users; for the rest, recommend_autoencoder returns empty → 0 hits
sample_users = pd.read_csv("data/eval_users.csv")["user_id"].tolist()
test_sample  = test[test["user_id"].isin(sample_users)]

print(f"Evaluating on {len(sample_users)} shared users (n=10) — AE trained on {len(sampled_train_users)} of {train['user_id'].nunique()} users...")

def recommend_fn(user_id, n):
    return recommend_autoencoder(user_id, model, matrix, train_sub, n=n)

metrics = evaluate(recommend_fn, test_sample, train_sub, n=10)
print(f"  Hit Rate : {metrics['hit_rate']}")
print(f"  Precision: {metrics['precision']}")
print(f"  Recall   : {metrics['recall']}")

# --- record run in tracker ---
# Edit COMMENT before each run to describe what changed.
COMMENT = (
    f"Dense 128→32→128, Sigmoid, masked MSE. "
    f"epochs={EPOCHS}, batch={BATCH_SIZE}, patience={PATIENCE}, train_users={TRAIN_USERS}."
)
log_run(
    "report/model_performance_tracker.xlsx",
    "Autoencoder",
    metrics,
    {
        "split": "leave-one-out", "min_ratings": 5, "n_recommendations": 10,
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "patience": PATIENCE,
        "train_users": TRAIN_USERS, "train_secs": TRAIN_SECS,
    },
    COMMENT,
)
print("✅ Results written to report/model_performance_tracker.xlsx")
