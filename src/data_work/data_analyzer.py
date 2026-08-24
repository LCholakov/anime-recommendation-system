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


def count_users(ratings_df: pd.DataFrame) -> int:
    return ratings_df["user_id"].nunique()


def avg_ratings_per_user(ratings_df: pd.DataFrame) -> float:
    return round(ratings_df.groupby("user_id").size().mean(), 2)


def top_10_most_rated_anime(ratings_df: pd.DataFrame, anime_df: pd.DataFrame) -> pd.DataFrame:
    counts = ratings_df.groupby("anime_id").agg(
        rating_count=("rating", "count"),
        avg_rating=("rating", "mean")
    ).reset_index()
    counts["avg_rating"] = counts["avg_rating"].round(2)
    result = counts.merge(anime_df[["anime_id", "name"]], on="anime_id")
    return result.nlargest(10, "rating_count")[["name", "rating_count", "avg_rating"]].reset_index(drop=True)


def top_10_users_by_most_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    counts = ratings_df.groupby("user_id").size().reset_index(name="rating_count")
    return counts.nlargest(10, "rating_count").reset_index(drop=True)


def top_10_users_by_least_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    counts = ratings_df.groupby("user_id").size().reset_index(name="rating_count")
    return counts.nsmallest(10, "rating_count").reset_index(drop=True)


def users_by_rating_count_brackets(ratings_df: pd.DataFrame) -> pd.DataFrame:
    counts = ratings_df.groupby("user_id").size()
    bins = [0, 3, 9, 50, 100, 500, 1000, 2500, float("inf")]
    labels = ["0–3", "4–9", "10–50", "51–100", "101–500", "501–1000", "1001–2500", "2500+"]
    brackets = pd.cut(counts, bins=bins, labels=labels, right=True)
    result = brackets.value_counts().reindex(labels).reset_index()
    result.columns = ["rating_count_range", "user_count"]
    return result
