import pandas as pd


def count_anime_titles(df: pd.DataFrame) -> int:
    return len(df)


def count_genres(df: pd.DataFrame) -> int:
    genres = df["genre"].dropna().str.split(", ")
    return genres.explode().nunique()


def list_genres(df: pd.DataFrame) -> list:
    genres = df["genre"].dropna().str.split(", ")
    return sorted(genres.explode().unique().tolist())


def list_types(df: pd.DataFrame) -> list:
    return sorted(df["type"].dropna().unique().tolist())


def top_10_by_rating(df: pd.DataFrame) -> pd.DataFrame:
    return df.nlargest(10, "rating")[["name", "genre", "type", "rating"]].reset_index(drop=True)


def top_10_by_members(df: pd.DataFrame) -> pd.DataFrame:
    return df.nlargest(10, "members")[["name", "genre", "type", "members"]].reset_index(drop=True)
