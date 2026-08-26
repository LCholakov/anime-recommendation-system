import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from src.models.baseline import compute_bayesian_scores, recommend_popular_anime
from src.models.bow import build_bow_matrix, recommend_bow
from src.models.tfidf import build_tfidf_matrix, recommend_tfidf
from src.models.svd import build_user_item_matrix as svd_build_matrix, train_svd, recommend_svd
from src.models.autoencoder import (
    build_user_item_matrix as ae_build_matrix,
    train_autoencoder, recommend_autoencoder, AnimeAutoencoder,
)
from src.models.ncf import NCF, encode_ids, train_ncf, recommend_ncf

SEED        = 42
N_USERS     = 5
N_RECS      = 10
OUTPUT_PATH = "report/manual_inspection.txt"
MODEL_DIR   = "model"

# ── load data ────────────────────────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv("data/train.csv")
test_df  = pd.read_csv("data/test.csv")
anime_df = pd.read_csv("data/anime_clean.csv")

import html as html_module
anime_df["name"] = anime_df["name"].apply(
    lambda x: html_module.unescape(x) if isinstance(x, str) else x
)
id_to_name = anime_df.set_index("anime_id")["name"].to_dict()

# ── sample users ─────────────────────────────────────────────────────────────
users_in_both = set(train_df["user_id"]) & set(test_df["user_id"])
sample_users  = pd.Series(sorted(users_in_both)).sample(N_USERS, random_state=SEED).tolist()

# ── build / load models ──────────────────────────────────────────────────────
import pickle

def _pkl_load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _pkl_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

# Baseline
print("Computing baseline scores...")
baseline_scores = compute_bayesian_scores(train_df)

# BoW
bow_pkl = os.path.join(MODEL_DIR, "bow_matrix.pkl")
if os.path.exists(bow_pkl):
    print("Loading BoW matrix from disk...")
    bow_matrix = _pkl_load(bow_pkl)
else:
    print("Building BoW matrix...")
    bow_matrix = build_bow_matrix(anime_df)
    _pkl_save(bow_matrix, bow_pkl)

# TF-IDF
tfidf_pkl = os.path.join(MODEL_DIR, "tfidf_matrix.pkl")
if os.path.exists(tfidf_pkl):
    print("Loading TF-IDF matrix from disk...")
    tfidf_matrix = _pkl_load(tfidf_pkl)
else:
    print("Building TF-IDF matrix...")
    tfidf_matrix = build_tfidf_matrix(anime_df)
    _pkl_save(tfidf_matrix, tfidf_pkl)

# SVD
svd_pkl = os.path.join(MODEL_DIR, "svd_reconstructed.pkl")
if os.path.exists(svd_pkl):
    print("Loading SVD reconstructed matrix from disk...")
    svd_recon, _, _ = _pkl_load(svd_pkl)
else:
    print("Training SVD...")
    svd_matrix = svd_build_matrix(train_df)
    svd_recon, Vt, anime_cols = train_svd(svd_matrix, n_components=50)
    _pkl_save((svd_recon, Vt, anime_cols), svd_pkl)

# Autoencoder
ae_pt  = os.path.join(MODEL_DIR, "autoencoder.pt")
ae_pkl = os.path.join(MODEL_DIR, "ae_matrix.pkl")
if os.path.exists(ae_pt) and os.path.exists(ae_pkl):
    print("Loading Autoencoder from disk...")
    ae_matrix = _pkl_load(ae_pkl)
    ae_model  = AnimeAutoencoder(ae_matrix.shape[1])
    ae_model.load_state_dict(torch.load(ae_pt, map_location="cpu", weights_only=True))
    ae_model.eval()
else:
    print("Training Autoencoder...")
    ae_matrix = ae_build_matrix(train_df)
    ae_model  = train_autoencoder(ae_matrix, epochs=20, batch_size=128, patience=3)
    torch.save(ae_model.state_dict(), ae_pt)
    _pkl_save(ae_matrix, ae_pkl)

# NCF
ncf_pt   = os.path.join(MODEL_DIR, "ncf.pt")
ncf_maps = os.path.join(MODEL_DIR, "ncf_maps.pkl")
if os.path.exists(ncf_pt) and os.path.exists(ncf_maps):
    print("Loading NCF from disk...")
    user_map, anime_map = _pkl_load(ncf_maps)
    ncf_model = NCF(len(user_map), len(anime_map), embed_dim=32)
    ncf_model.load_state_dict(torch.load(ncf_pt, map_location="cpu", weights_only=True))
    ncf_model.eval()
else:
    print("Training NCF...")
    ncf_model, user_map, anime_map = train_ncf(
        train_df, embed_dim=32, epochs=20, batch_size=256, patience=3
    )
    torch.save(ncf_model.state_dict(), ncf_pt)
    _pkl_save((user_map, anime_map), ncf_maps)

# ── report ───────────────────────────────────────────────────────────────────
MODELS = ["Baseline", "BoW", "TF-IDF", "SVD", "Autoencoder", "NCF"]

lines = []
lines.append("=" * 70)
lines.append("ALL-MODELS MANUAL INSPECTION REPORT")
lines.append(f"Seed: {SEED} | Users sampled: {N_USERS} | Recommendations per model: {N_RECS}")
lines.append("=" * 70)

for user_id in sample_users:
    train_user = train_df[train_df["user_id"] == user_id].sort_values("rating", ascending=False)
    test_user  = test_df[test_df["user_id"] == user_id]
    test_ids   = set(test_user["anime_id"])

    lines.append(f"\nUser {user_id}")
    lines.append("-" * 40)
    lines.append("Training favourites (top 5 by rating):")
    for _, row in train_user.head(5).iterrows():
        lines.append(f"  [{row['rating']:4.1f}] {id_to_name.get(row['anime_id'], str(row['anime_id']))}")

    # collect recommendations per model
    model_recs = {
        "Baseline":   recommend_popular_anime(baseline_scores, train_df, user_id=user_id, n=N_RECS),
        "BoW":        recommend_bow(user_id, train_df, bow_matrix, anime_df, n=N_RECS),
        "TF-IDF":     recommend_tfidf(user_id, train_df, tfidf_matrix, anime_df, n=N_RECS),
        "SVD":        recommend_svd(user_id, svd_recon, train_df, n=N_RECS),
        "Autoencoder":recommend_autoencoder(user_id, ae_model, ae_matrix, train_df, n=N_RECS),
        "NCF":        recommend_ncf(user_id, ncf_model, user_map, anime_map, train_df, n=N_RECS)
                      if user_id in user_map
                      else pd.DataFrame(columns=["anime_id"]),
    }

    # score column per model
    score_col = {
        "Baseline": "bayesian_score", "BoW": "bow_score", "TF-IDF": "tfidf_score",
        "SVD": "predicted_rating", "Autoencoder": "predicted_score", "NCF": "predicted_rating",
    }

    for model_name in MODELS:
        recs    = model_recs[model_name]
        rec_ids = set(recs["anime_id"]) if not recs.empty else set()
        hits    = rec_ids & test_ids
        scol    = score_col[model_name]

        lines.append(f"\n  [{model_name}]  hits: {len(hits)}/{len(test_ids)}")
        if recs.empty:
            lines.append("    (no recommendations — user not in model)")
            continue
        for rank, (_, row) in enumerate(recs.iterrows(), 1):
            aid  = row["anime_id"]
            name = id_to_name.get(aid, str(aid))
            sc   = f"{row[scol]:.4f}" if scol in row and pd.notna(row[scol]) else "N/A"
            hit  = " ✓" if aid in hits else ""
            lines.append(f"    {rank:2}. {name} (score: {sc}){hit}")

    lines.append("\n  Hidden relevant titles (test set):")
    for _, row in test_user.iterrows():
        name = id_to_name.get(row["anime_id"], str(row["anime_id"]))
        lines.append(f"    [{row['rating']:4.1f}] {name}")

    lines.append("=" * 70)

output = "\n".join(lines)
os.makedirs("report", exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(output)

print(output)
print(f"\n✅ Report saved to {OUTPUT_PATH}")
