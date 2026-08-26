import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from src.models.ncf import train_ncf, recommend_ncf
from src.models.evaluator import evaluate, append_to_tracker

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
model, user_map, anime_map = train_ncf(
    train,
    embed_dim=EMBED_DIM,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    patience=PATIENCE,
)
print(f"  Users: {len(user_map)} | Anime: {len(anime_map)}")

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

# --- write to model performance tracker ---
HEADERS = [
    "Model", "min_ratings", "test_ratio", "n_recommendations",
    "m (percentile)", "Hit Rate @10", "Precision @10", "Recall @10", "Comments"
]
row_data = [
    "NCF (Neural Collaborative Filtering)",
    5, "leave-one-out", 10, "N/A",
    metrics["hit_rate"], metrics["precision"], metrics["recall"],
    f"User+anime embeddings ({EMBED_DIM}d), concat→Dense64→Dense32→1. "
    f"MSE loss, Adam, epochs={EPOCHS}, batch={BATCH_SIZE}, patience={PATIENCE}."
]
append_to_tracker("report/model_performance_tracker.xlsx", row_data, HEADERS)
print("✅ Results written to report/model_performance_tracker.xlsx")
