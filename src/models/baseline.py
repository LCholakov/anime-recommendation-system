import pandas as pd


def split_train_test(ratings_df: pd.DataFrame, min_ratings: int = 5, test_ratio: float = 0.3) -> tuple:
    counts = ratings_df.groupby("user_id").size()
    valid_users = counts[counts >= min_ratings].index
    df = ratings_df[ratings_df["user_id"].isin(valid_users)].copy()

    train_rows = []
    test_rows = []
    for _, group in df.groupby("user_id"):
        group = group.sample(frac=1, random_state=42)
        n_test = max(1, int(len(group) * test_ratio))
        test_rows.append(group.iloc[:n_test])
        train_rows.append(group.iloc[n_test:])

    train = pd.concat(train_rows).reset_index(drop=True)
    test = pd.concat(test_rows).reset_index(drop=True)
    return train, test


def compute_bayesian_scores(train_df: pd.DataFrame) -> pd.DataFrame:
    stats = train_df.groupby("anime_id")["rating"].agg(
        v="count",
        R="mean"
    ).reset_index()

    C = train_df["rating"].mean()
    m = stats["v"].quantile(0.8)

    stats["bayesian_score"] = (
        (stats["v"] / (stats["v"] + m)) * stats["R"] +
        (m / (stats["v"] + m)) * C
    )
    return stats[["anime_id", "v", "R", "bayesian_score"]].rename(
        columns={"v": "rating_count", "R": "avg_rating"}
    )


def recommend_popular_anime(
    scores_df: pd.DataFrame,
    ratings_df: pd.DataFrame = None,
    user_id: int = None,
    n: int = 10
) -> pd.DataFrame:
    df = scores_df.copy()
    if ratings_df is not None and user_id is not None:
        already_rated = ratings_df[ratings_df["user_id"] == user_id]["anime_id"]
        df = df[~df["anime_id"].isin(already_rated)]
    return df.nlargest(n, "bayesian_score").reset_index(drop=True)


def evaluate_model(
    scores_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    n: int = 10
) -> dict:
    precisions, recalls, hits = [], [], []

    for user_id, test_group in test_df.groupby("user_id"):
        recs = recommend_popular_anime(scores_df, train_df, user_id=user_id, n=n)
        rec_ids = set(recs["anime_id"])
        test_ids = set(test_group["anime_id"])

        hit = len(rec_ids & test_ids) > 0
        precision = len(rec_ids & test_ids) / n
        recall = len(rec_ids & test_ids) / len(test_ids) if test_ids else 0.0

        hits.append(hit)
        precisions.append(precision)
        recalls.append(recall)

    return {
        "hit_rate":  round(sum(hits) / len(hits), 4),
        "precision": round(sum(precisions) / len(precisions), 4),
        "recall":    round(sum(recalls) / len(recalls), 4),
    }
