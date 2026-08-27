import sys
import os
import time
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from src.models.ncf import train_ncf, recommend_ncf
from src.models.evaluator import evaluate, log_run

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

SEED        = 42
EPOCHS      = 20
BATCH_SIZE  = 256
PATIENCE    = 3
EMBED_DIM   = 32
N_PER_USER  = 10      # BPR positive pairs sampled per user per epoch
EVAL_USERS  = 1000

torch.manual_seed(SEED)

# --- load data ---
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")
print(f"Train: {len(train)} rows | Test: {len(test)} rows")

# --- train NCF ---
print(f"Training NCF (embed={EMBED_DIM}, epochs={EPOCHS}, batch={BATCH_SIZE}, patience={PATIENCE}, n_per_user={N_PER_USER})...")
_t0 = time.time()
model, user_map, anime_map = train_ncf(
    train,
    embed_dim=EMBED_DIM,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    patience=PATIENCE,
    n_per_user=N_PER_USER,
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

# --- save model artifacts ---
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "ncf.pt"))
with open(os.path.join(MODEL_DIR, "ncf_maps.pkl"), "wb") as f:
    pickle.dump((user_map, anime_map), f)
print("✅ Saved model/ncf.pt + model/ncf_maps.pkl")

# --- record run in tracker ---
# Edit COMMENT before each run to describe what changed.
COMMENT = (
    f"NCF v2 — User+anime embeddings ({EMBED_DIM}d), concat→Dense64→Dense32→1. "
    f"BPR ranking loss, Adam. epochs={EPOCHS}, batch={BATCH_SIZE}, "
    f"patience={PATIENCE}, n_per_user={N_PER_USER}."
)
log_run(
    "report/model_performance_tracker.xlsx",
    "NCF (Neural Collaborative Filtering)",
    metrics,
    {
        "split": "leave-one-out", "min_ratings": 5, "n_recommendations": 10,
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "patience": PATIENCE,
        "embed_dim": EMBED_DIM, "n_per_user": N_PER_USER, "train_secs": TRAIN_SECS,
    },
    COMMENT,
)
print("✅ Results written to report/model_performance_tracker.xlsx")
