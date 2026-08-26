import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD


def build_user_item_matrix(train_df: pd.DataFrame) -> pd.DataFrame:
    matrix = train_df.pivot_table(
        index="user_id", columns="anime_id", values="rating", aggfunc="mean"
    ).fillna(0)
    return matrix


def train_svd(matrix: pd.DataFrame, n_components: int = 50) -> pd.DataFrame:
    n_components = min(n_components, min(matrix.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U = svd.fit_transform(matrix.values)   # (n_users, n_components)
    Vt = svd.components_                   # (n_components, n_anime)
    reconstructed = U @ Vt                 # (n_users, n_anime)
    return pd.DataFrame(reconstructed, index=matrix.index, columns=matrix.columns)


def recommend_svd(
    user_id: int,
    reconstructed: pd.DataFrame,
    train_df: pd.DataFrame,
    n: int = 10
) -> pd.DataFrame:
    if user_id not in reconstructed.index:
        return pd.DataFrame(columns=["anime_id", "predicted_rating"])

    rated_ids = set(train_df[train_df["user_id"] == user_id]["anime_id"])
    scores    = reconstructed.loc[user_id].copy()
    scores    = scores.drop(index=[i for i in rated_ids if i in scores.index])

    top = scores.nlargest(n).reset_index()
    top.columns = ["anime_id", "predicted_rating"]
    return top
