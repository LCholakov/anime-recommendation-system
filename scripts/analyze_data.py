import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_work.data_analyzer import (
    count_anime_titles,
    count_genres,
    list_genres,
    list_types,
    top_10_by_rating,
    top_10_by_members,
)

df = pd.read_csv("data/anime_clean.csv")

titles     = count_anime_titles(df)
genre_count = count_genres(df)
genres     = list_genres(df)
types      = list_types(df)
top_rating = top_10_by_rating(df)
top_members = top_10_by_members(df)

# --- terminal output ---
print(f"Total anime titles : {titles}")
print(f"Total unique genres: {genre_count}")
print(f"\nGenres: {', '.join(genres)}")
print(f"\nTypes: {', '.join(types)}")
print("\n--- Top 10 by Rating ---")
print(top_rating.to_string(index=False))
print("\n--- Top 10 by Members ---")
print(top_members.to_string(index=False))

