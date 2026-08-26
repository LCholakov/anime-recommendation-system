import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def build_bow_matrix(anime_df: pd.DataFrame) -> pd.DataFrame:
    genres = anime_df["genre"].fillna("").str.lower().str.replace(",", " ")
    all_words = sorted(set(
        word.strip()
        for cell in genres
        for word in cell.split()
        if word.strip()
    ))
    rows = []
    for _, row in anime_df.iterrows():
        words = set(row["genre"].lower().replace(",", " ").split()) if pd.notna(row["genre"]) else set()
        rows.append({w: 1 for w in words if w.strip() in all_words})
    raw = pd.DataFrame(rows, index=anime_df["anime_id"], columns=all_words).fillna(0).astype(np.float32)
    # drop zero-norm rows (anime with no recognisable genre words)
    norms = np.linalg.norm(raw.values, axis=1)
    raw   = raw[norms > 0]
    # pre-normalise rows to unit length — keeps matmul in [0,1] and avoids overflow
    normalised = normalize(raw.values.astype(np.float32), norm="l2")
    return pd.DataFrame(normalised, index=raw.index, columns=raw.columns)


def get_similar_anime(
    anime_id: int,
    bow_matrix: pd.DataFrame,
    anime_df: pd.DataFrame,
    n: int = 10
) -> pd.DataFrame:
    # rows are already unit-normalised — dot product == cosine similarity
    vec  = bow_matrix.loc[[anime_id]].values
    sims = (bow_matrix.values @ vec.T).flatten()

    result = pd.DataFrame({"anime_id": bow_matrix.index, "similarity": sims})
    result = result[result["anime_id"] != anime_id]
    result = result.sort_values("similarity", ascending=False).head(n).reset_index(drop=True)
    result = result.merge(anime_df[["anime_id", "name", "genre"]], on="anime_id", how="left")
    return result


def recommend_bow(
    user_id: int,
    train_df: pd.DataFrame,
    bow_matrix: pd.DataFrame,
    anime_df: pd.DataFrame,
    n: int = 10
) -> pd.DataFrame:
    user_ratings = train_df[train_df["user_id"] == user_id].copy()
    rated_ids    = set(user_ratings["anime_id"])

    valid = user_ratings[user_ratings["anime_id"].isin(bow_matrix.index)]
    if valid.empty:
        return pd.DataFrame(columns=["anime_id", "bow_score", "name", "genre"])

    # rows already unit-normalised — dot product == cosine similarity (float32 throughout)
    user_vecs   = bow_matrix.loc[valid["anime_id"]].values         # already float32
    ratings_arr = valid["rating"].values.astype(np.float32)
    all_vecs    = bow_matrix.values                                 # already float32

    sims     = user_vecs @ all_vecs.T                              # (n_rated, n_anime)
    weighted = (sims * ratings_arr[:, np.newaxis]).sum(axis=0)     # (n_anime,)
    weighted = np.nan_to_num(weighted, nan=0.0, posinf=0.0, neginf=0.0)

    score_series = pd.Series(weighted, index=bow_matrix.index)
    score_series = score_series.drop(index=[i for i in rated_ids if i in score_series.index])
    top = score_series.nlargest(n).reset_index()
    top.columns = ["anime_id", "bow_score"]
    top = top.merge(anime_df[["anime_id", "name", "genre"]], on="anime_id", how="left")
    return top
