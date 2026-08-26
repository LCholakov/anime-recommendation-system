import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def build_tfidf_matrix(anime_df: pd.DataFrame) -> pd.DataFrame:
    genres = anime_df["genre"].fillna("").str.lower().str.replace(",", " ")

    vectorizer = TfidfVectorizer(token_pattern=r"[a-z][a-z\-]+")
    tfidf = vectorizer.fit_transform(genres).toarray().astype(np.float64)

    # drop zero-norm rows (anime with no recognisable genre words)
    norms = np.linalg.norm(tfidf, axis=1)
    mask  = norms > 0
    tfidf = tfidf[mask]
    ids   = anime_df["anime_id"].values[mask]

    # pre-normalise to unit length — dot product then equals cosine similarity
    tfidf = normalize(tfidf, norm="l2")

    return pd.DataFrame(tfidf, index=ids, columns=vectorizer.get_feature_names_out())


def get_similar_anime(
    anime_id: int,
    tfidf_matrix: pd.DataFrame,
    anime_df: pd.DataFrame,
    n: int = 10
) -> pd.DataFrame:
    vec  = tfidf_matrix.loc[[anime_id]].values
    sims = (tfidf_matrix.values @ vec.T).flatten()

    result = pd.DataFrame({"anime_id": tfidf_matrix.index, "similarity": sims})
    result = result[result["anime_id"] != anime_id]
    result = result.sort_values("similarity", ascending=False).head(n).reset_index(drop=True)
    result = result.merge(anime_df[["anime_id", "name", "genre"]], on="anime_id", how="left")
    return result


def recommend_tfidf(
    user_id: int,
    train_df: pd.DataFrame,
    tfidf_matrix: pd.DataFrame,
    anime_df: pd.DataFrame,
    n: int = 10
) -> pd.DataFrame:
    user_ratings = train_df[train_df["user_id"] == user_id].copy()
    rated_ids    = set(user_ratings["anime_id"])

    valid = user_ratings[user_ratings["anime_id"].isin(tfidf_matrix.index)]
    if valid.empty:
        return pd.DataFrame(columns=["anime_id", "tfidf_score", "name", "genre"])

    # cast to float64 for the matmul — float32 overflows on large matrices
    user_vecs   = tfidf_matrix.loc[valid["anime_id"]].values.astype(np.float64)
    ratings_arr = valid["rating"].values.astype(np.float64)
    all_vecs    = tfidf_matrix.values.astype(np.float64)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        sims = user_vecs @ all_vecs.T                              # (n_rated, n_anime)
    weighted = (sims * ratings_arr[:, np.newaxis]).sum(axis=0)    # (n_anime,)
    weighted = np.nan_to_num(weighted, nan=0.0, posinf=0.0, neginf=0.0)

    score_series = pd.Series(weighted, index=tfidf_matrix.index)
    score_series = score_series.drop(index=[i for i in rated_ids if i in score_series.index])
    top = score_series.nlargest(n).reset_index()
    top.columns = ["anime_id", "tfidf_score"]
    top = top.merge(anime_df[["anime_id", "name", "genre"]], on="anime_id", how="left")
    return top
