import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD


def build_user_item_matrix(train_df: pd.DataFrame) -> pd.DataFrame:
    matrix = train_df.pivot_table(
        index="user_id", columns="anime_id", values="rating", aggfunc="mean"
    ).fillna(0)
    return matrix


def train_svd(matrix: pd.DataFrame, n_components: int = 50) -> tuple:
    """Returns (reconstructed_df, Vt, anime_columns) so new users can be folded in."""
    n_components = min(n_components, min(matrix.shape) - 1)
    # float64 — float32 overflows in sklearn's randomised SVD on large sparse matrices
    values = matrix.values.astype(np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        U = svd.fit_transform(values)      # (n_users, n_components)
    Vt = svd.components_                   # (n_components, n_anime)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        reconstructed = U @ Vt             # (n_users, n_anime)
    recon_df = pd.DataFrame(reconstructed, index=matrix.index, columns=matrix.columns)
    return recon_df, Vt, matrix.columns.tolist()


def fold_in_user(
    picks: list,
    Vt: np.ndarray,
    anime_columns: list,
    n: int = 10,
) -> pd.DataFrame:
    """Score all anime for a new user by projecting their ratings onto Vt.

    picks: list of {"anime_id": int, "rating": float}
    Returns top-n as DataFrame with columns ["anime_id", "predicted_rating"].
    """
    col_index = {aid: i for i, aid in enumerate(anime_columns)}
    rated_ids = set()
    r = np.zeros(len(anime_columns), dtype=np.float32)
    for p in picks:
        aid = p["anime_id"]
        if aid in col_index:
            r[col_index[aid]] = float(p["rating"])
            rated_ids.add(aid)

    # project: u_latent = r @ Vt^T  (shape: n_components)
    u_latent = r @ Vt.T                     # (n_components,)
    scores   = u_latent @ Vt                # (n_anime,)
    scores   = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    score_series = pd.Series(scores, index=anime_columns)
    score_series = score_series.drop(index=[i for i in rated_ids if i in score_series.index])
    top = score_series.nlargest(n).reset_index()
    top.columns = ["anime_id", "predicted_rating"]
    return top


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
