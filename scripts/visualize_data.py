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
    count_users,
    avg_ratings_per_user,
    top_10_most_rated_anime,
    top_10_users_by_most_ratings,
    users_by_rating_count_brackets,
)

CSS = """
  body { font-family: system-ui, sans-serif; max-width: 860px; margin: 40px auto; padding: 0 20px; color: #1f2328; }
  h1   { font-size: 1.6rem; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; }
  h2   { font-size: 1.1rem; margin-top: 32px; color: #3b82d4; }
  .stats { display: flex; gap: 32px; margin: 16px 0; }
  .stat  { background: #f7f8fa; border: 1px solid #e5e7eb; border-radius: 6px; padding: 16px 24px; }
  .stat span { display: block; font-size: 2rem; font-weight: 700; }
  .stat label { font-size: 0.85rem; color: #57606a; }
  .tags { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
  .tag  { background: #f7f8fa; border: 1px solid #e5e7eb; border-radius: 4px; padding: 2px 10px; font-size: 0.82rem; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.9rem; }
  th    { background: #f7f8fa; border-bottom: 2px solid #e5e7eb; text-align: left; padding: 8px 10px; }
  td    { border-bottom: 1px solid #e5e7eb; padding: 7px 10px; }
  tr:last-child td { border-bottom: none; }
"""

# --- anime analysis ---
anime_df = pd.read_csv("data/anime_clean.csv")

titles      = count_anime_titles(anime_df)
genre_count = count_genres(anime_df)
genres      = list_genres(anime_df)
types       = list_types(anime_df)
top_rating  = top_10_by_rating(anime_df)
top_members = top_10_by_members(anime_df)

anime_html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Anime Analysis</title><style>{CSS}</style></head>
<body>
<h1>Anime Data Analysis</h1>
<div class="stats">
  <div class="stat"><span>{titles}</span><label>Anime titles</label></div>
  <div class="stat"><span>{genre_count}</span><label>Unique genres</label></div>
  <div class="stat"><span>{len(types)}</span><label>Types</label></div>
</div>
<h2>Genres</h2>
<div class="tags">{''.join(f'<span class="tag">{g}</span>' for g in genres)}</div>
<h2>Types</h2>
<div class="tags">{''.join(f'<span class="tag">{t}</span>' for t in types)}</div>
<h2>Top 10 by Rating</h2>
{top_rating.to_html(index=False, border=0)}
<h2>Top 10 by Members</h2>
{top_members.to_html(index=False, border=0)}
</body></html>"""

with open("data/anime_analysis.html", "w", encoding="utf-8") as f:
    f.write(anime_html)
print("✅ Saved to: data/anime_analysis.html")

# --- ratings analysis ---
ratings_df = pd.read_csv("data/rating_clean.csv")

n_users        = count_users(ratings_df)
avg_per_user   = avg_ratings_per_user(ratings_df)
most_rated     = top_10_most_rated_anime(ratings_df, anime_df)
top_raters     = top_10_users_by_most_ratings(ratings_df)
rating_brackets = users_by_rating_count_brackets(ratings_df)
top_avg_rating = top_10_by_rating(anime_df)

ratings_html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Ratings Analysis</title><style>{CSS}</style></head>
<body>
<h1>Ratings Data Analysis</h1>
<div class="stats">
  <div class="stat"><span>{n_users}</span><label>Unique users</label></div>
  <div class="stat"><span>{avg_per_user}</span><label>Avg ratings per user</label></div>
</div>
<h2>Top 10 Most Rated Anime</h2>
{most_rated.to_html(index=False, border=0)}
<h2>Top 10 Highest Rated Anime</h2>
{top_avg_rating.to_html(index=False, border=0)}
<h2>Top 10 Users by Most Ratings</h2>
{top_raters.to_html(index=False, border=0)}
<h2>Users by Rating Count</h2>
{rating_brackets.to_html(index=False, border=0)}
</body></html>"""

with open("data/ratings_analysis.html", "w", encoding="utf-8") as f:
    f.write(ratings_html)
print("✅ Saved to: data/ratings_analysis.html")
