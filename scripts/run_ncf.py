import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from src.models.ncf import train_ncf, recommend_ncf
from src.models.evaluator import evaluate, log_run

SEED       = 42
EPOCHS     = 20
BATCH_SIZE = 256
PATIENCE   = 3
EMBED_DIM  = 32
EVAL_USERS = 1000

torch.manual_seed(SEED)

# --- load data ---
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")
print(f"Train: {len(train)} rows | Test: {len(test)} rows")

# --- train NCF ---
print(f"Training NCF (embed={EMBED_DIM}, epochs={EPOCHS}, batch={BATCH_SIZE}, patience={PATIENCE})...")
_t0 = time.time()
model, user_map, anime_map = train_ncf(
    train,
    embed_dim=EMBED_DIM,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    patience=PATIENCE,
)
TRAIN_SECS = round(time.time() - _t0, 1)
print(f"  Users: {len(user_map)} | Anime: {len(anime_map)}")
print(f"  Train time: {TRAIN_SECS}s")

# --- shared eval sample (same 1000 users across all models) ---
# NCF can only score users in user_map; others get empty recs → 0 hits
sample_users = pd.read_csv("data/eval_users.csv")["user_id"].tolist()
test_sample  = test[test["user_id"].isin(sample_users)]

print(f"Evaluating on {len(sample_users)} shared users (n=10) — NCF covers {len(user_map)} of {train['user_id'].nunique()} users...")
metrics = evaluate(
    lambda user_id, n: recommend_ncf(user_id, model, user_map, anime_map, train, n=n),
    test_sample, train, n=10
)
print(f"  Hit Rate : {metrics['hit_rate']}")
print(f"  Precision: {metrics['precision']}")
print(f"  Recall   : {metrics['recall']}")

# --- record run in tracker ---
# Edit COMMENT before each run to describe what changed.
COMMENT = (
    f"User+anime embeddings ({EMBED_DIM}d), concat→Dense64→Dense32→1. "
    f"MSE loss, Adam. epochs={EPOCHS}, batch={BATCH_SIZE}, patience={PATIENCE}."
)
log_run(
    "report/model_performance_tracker.xlsx",
    "NCF (Neural Collaborative Filtering)",
    metrics,
    {
        "split": "leave-one-out", "min_ratings": 5, "n_recommendations": 10,
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "patience": PATIENCE,
        "embed_dim": EMBED_DIM, "train_secs": TRAIN_SECS,
    },
    COMMENT,
)
print("✅ Results written to report/model_performance_tracker.xlsx")
