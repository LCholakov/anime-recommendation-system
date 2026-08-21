import pandas as pd


def clean_rating_data(df: pd.DataFrame, valid_anime_ids: pd.Series) -> pd.DataFrame:
    original_count = len(df)
    df = df.dropna(subset=["user_id", "anime_id"])
    df = df.drop_duplicates(subset=["user_id", "anime_id"])
    df = df[df["rating"].between(1, 10)]
    df = df[df["anime_id"].isin(valid_anime_ids)]
    removed_count = original_count - len(df)
    pct_removed = removed_count / original_count * 100
    print(f"✅ Ratings cleaned ({len(df)} rows kept, {removed_count} rows removed, {pct_removed:.1f}% of total)")
    return df


def clean_anime_data(df: pd.DataFrame) -> pd.DataFrame:
    original_count = len(df)
    df = df.drop_duplicates(subset="anime_id")
    df = df.dropna(subset=["anime_id"])
    df = df.dropna()
    removed_count = original_count - len(df)
    pct_removed = removed_count / original_count * 100
    print(f"✅ Anime cleaned ({len(df)} rows kept, {removed_count} rows removed, {pct_removed:.1f}% of total)")
    return df


def save_dataframe(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False)
    print(f"✅ Saved to: {output_path}")
