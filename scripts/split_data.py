"""Regenerate train.csv and test.csv using leave-one-out split.

Run from project root:
    python scripts/split_data.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_work.data_loader import load_rating_data
from src.models.baseline import split_train_test

print("Loading cleaned ratings...")
ratings_df = load_rating_data("data/rating_clean.csv")

print("Splitting (leave-one-out, min_ratings=5)...")
train, test = split_train_test(ratings_df, min_ratings=5)

print(f"Train: {len(train):,} rows | {train['user_id'].nunique():,} users")
print(f"Test:  {len(test):,} rows  | {test['user_id'].nunique():,} users")
print(f"Test items per user: always exactly 1 (leave-one-out)")

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
print("✅ Saved data/train.csv and data/test.csv")
