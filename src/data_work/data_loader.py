import pandas as pd


def load_anime_data(path: str = "data/anime.csv") -> pd.DataFrame:
    # pass
    return pd.read_csv(path)


def load_rating_data(path: str = "data/rating.csv") -> pd.DataFrame:
    # pass
    return pd.read_csv(path)
