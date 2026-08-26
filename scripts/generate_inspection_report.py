import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.models.baseline import recommend_popular_anime

SEED        = 42
N_USERS     = 5
N_RECS      = 10
OUTPUT_PATH = "report/manual_inspection.txt"

# --- load pre-computed data ---
train_df  = pd.read_csv("data/train.csv")
test_df   = pd.read_csv("data/test.csv")
scores_df = pd.read_csv("data/baseline_scores.csv")
anime_df  = pd.read_csv("data/anime_clean.csv")

# map anime_id -> name
id_to_name = anime_df.set_index("anime_id")["name"].to_dict()

# pick 5 users who appear in both train and test
users_in_both = set(train_df["user_id"]) & set(test_df["user_id"])
sample_users  = sorted(users_in_both)
rng           = pd.Series(list(sample_users)).sample(N_USERS, random_state=SEED).tolist()

lines = []
lines.append("=" * 70)
lines.append("BASELINE MODEL — MANUAL INSPECTION REPORT")
lines.append(f"Seed: {SEED} | Users sampled: {N_USERS} | Recommendations: {N_RECS}")
lines.append("=" * 70)

for user_id in rng:
    recs    = recommend_popular_anime(scores_df, train_df, user_id=user_id, n=N_RECS)
    rec_ids = set(recs["anime_id"])

    train_items = train_df[train_df["user_id"] == user_id].sort_values("rating", ascending=False)
    test_items  = test_df[test_df["user_id"] == user_id]
    test_ids    = set(test_items["anime_id"])
    hits        = rec_ids & test_ids

    lines.append(f"\nUser {user_id}")
    lines.append("-" * 40)

    lines.append("Training favourites (top 5 by rating):")
    for _, row in train_items.head(5).iterrows():
        name = id_to_name.get(row["anime_id"], f"ID {row['anime_id']}")
        lines.append(f"  [{row['rating']:4.1f}] {name}")

    lines.append("\nRecommendations:")
    for i, (_, row) in enumerate(recs.iterrows(), 1):
        name  = id_to_name.get(row["anime_id"], f"ID {row['anime_id']}")
        hit   = "✓ HIT" if row["anime_id"] in hits else ""
        lines.append(f"  {i:2}. {name} (score: {row['bayesian_score']:.3f}) {hit}")

    lines.append("\nHidden relevant titles (test set):")
    for _, row in test_items.iterrows():
        name   = id_to_name.get(row["anime_id"], f"ID {row['anime_id']}")
        marker = "✓ found" if row["anime_id"] in hits else "✗ missed"
        lines.append(f"  [{row['rating']:4.1f}] {name} — {marker}")

    lines.append(f"\nHits: {len(hits)}/{len(test_ids)}")
    lines.append("=" * 70)

output = "\n".join(lines)
os.makedirs("report", exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(output)

print(output)
print(f"\n✅ Report saved to {OUTPUT_PATH}")
