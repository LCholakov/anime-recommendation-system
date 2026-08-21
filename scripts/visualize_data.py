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

titles      = count_anime_titles(df)
genre_count = count_genres(df)
genres      = list_genres(df)
types       = list_types(df)
top_rating  = top_10_by_rating(df)
top_members = top_10_by_members(df)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Anime Data Analysis</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 860px; margin: 40px auto; padding: 0 20px; color: #1f2328; }}
  h1   {{ font-size: 1.6rem; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; }}
  h2   {{ font-size: 1.1rem; margin-top: 32px; color: #3b82d4; }}
  .stats {{ display: flex; gap: 32px; margin: 16px 0; }}
  .stat  {{ background: #f7f8fa; border: 1px solid #e5e7eb; border-radius: 6px; padding: 16px 24px; }}
  .stat span {{ display: block; font-size: 2rem; font-weight: 700; }}
  .stat label {{ font-size: 0.85rem; color: #57606a; }}
  .tags {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }}
  .tag  {{ background: #f7f8fa; border: 1px solid #e5e7eb; border-radius: 4px; padding: 2px 10px; font-size: 0.82rem; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.9rem; }}
  th    {{ background: #f7f8fa; border-bottom: 2px solid #e5e7eb; text-align: left; padding: 8px 10px; }}
  td    {{ border-bottom: 1px solid #e5e7eb; padding: 7px 10px; }}
  tr:last-child td {{ border-bottom: none; }}
</style>
</head>
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

</body>
</html>"""

output_path = "data/anime_analysis.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"✅ HTML report saved to: {output_path}")
