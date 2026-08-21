import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_work.data_loader import load_anime_data, load_rating_data
from src.data_work.data_cleaner import clean_anime_data, clean_rating_data, save_dataframe

anime_df = load_anime_data()
clean_anime = clean_anime_data(anime_df)
save_dataframe(clean_anime, "data/anime_clean.csv")

rating_df = load_rating_data()
clean_ratings = clean_rating_data(rating_df, clean_anime["anime_id"])
save_dataframe(clean_ratings, "data/rating_clean.csv")
