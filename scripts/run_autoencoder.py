import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from src.data_work.data_loader import load_anime_data
from src.models.autoencoder import build_user_item_matrix, train_autoencoder, recommend_autoencoder
from src.models.evaluator import evaluate, append_to_tracker

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
model = train_autoencoder(matrix, epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE)

# --- evaluate on a sample ---
test_users   = test["user_id"].unique()
sample_users = pd.Series(test_users).sample(min(EVAL_USERS, len(test_users)), random_state=SEED).tolist()
sample_users = [u for u in sample_users if u in matrix.index]
test_sample  = test[test["user_id"].isin(sample_users)]

print(f"Evaluating on {len(sample_users)} sampled users (n=10, relevant=rating>=7)...")

def recommend_fn(user_id, n):
    return recommend_autoencoder(user_id, model, matrix, train_sub, n=n)

metrics = evaluate(recommend_fn, test_sample, train_sub, n=10)
print(f"  Hit Rate : {metrics['hit_rate']}")
print(f"  Precision: {metrics['precision']}")
print(f"  Recall   : {metrics['recall']}")

# --- write to model performance tracker ---
HEADERS = [
    "Model", "min_ratings", "test_ratio", "n_recommendations",
    "m (percentile)", "Hit Rate @10", "Precision @10", "Recall @10", "Comments"
]
row_data = [
    "Autoencoder",
    5, 0.3, 10, "N/A",
    metrics["hit_rate"], metrics["precision"], metrics["recall"],
    f"Collaborative AE. Dense 128→32→128, Sigmoid output, masked MSE loss. "
    f"Adam, epochs={EPOCHS}, batch={BATCH_SIZE}, patience={PATIENCE}, train_users={TRAIN_USERS}."
]
append_to_tracker("report/model_performance_tracker.xlsx", row_data, HEADERS)
print("✅ Results written to report/model_performance_tracker.xlsx")
