import sys
import os
import html as html_module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎌",
    layout="wide",
)

# ── paths ────────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANIME_CSV    = os.path.join(ROOT, "data", "anime_clean.csv")
TRAIN_CSV    = os.path.join(ROOT, "data", "train.csv")
TRACKER_XLSX = os.path.join(ROOT, "report", "model_performance_tracker.xlsx")
ANIME_HTML   = os.path.join(ROOT, "data", "anime_analysis.html")
RATINGS_HTML = os.path.join(ROOT, "data", "ratings_analysis.html")


# ── cached data loaders ──────────────────────────────────────────────────────
@st.cache_data
def load_anime() -> pd.DataFrame:
    from src.data_work.data_loader import load_anime_data
    df = load_anime_data(ANIME_CSV)
    # decode HTML entities stored in the CSV (e.g. &quot; → ")
    df["name"] = df["name"].apply(lambda x: html_module.unescape(x) if isinstance(x, str) else x)
    return df


@st.cache_data
def load_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV)


@st.cache_data
def load_tracker() -> pd.DataFrame:
    df = pd.read_excel(TRACKER_XLSX)
    # keep only unique model rows (first occurrence per Model name)
    return df.drop_duplicates(subset=["Model"], keep="first").reset_index(drop=True)


# ── cached model builders ────────────────────────────────────────────────────
@st.cache_resource
def get_bow_matrix():
    from src.models.bow import build_bow_matrix
    return build_bow_matrix(load_anime())


@st.cache_resource
def get_tfidf_matrix():
    from src.models.tfidf import build_tfidf_matrix
    return build_tfidf_matrix(load_anime())


@st.cache_resource
def get_svd_reconstructed():
    from src.models.svd import build_user_item_matrix, train_svd
    train = load_train()
    matrix = build_user_item_matrix(train)
    return train_svd(matrix, n_components=50)


@st.cache_resource
def get_autoencoder():
    from src.models.autoencoder import build_user_item_matrix, train_autoencoder
    train = load_train()
    matrix = build_user_item_matrix(train)
    model = train_autoencoder(matrix, epochs=20, batch_size=128, patience=3)
    return model, matrix


@st.cache_resource
def get_ncf():
    from src.models.ncf import train_ncf
    train = load_train()
    model, user_map, anime_map = train_ncf(
        train, embed_dim=32, epochs=20, batch_size=256, patience=3
    )
    return model, user_map, anime_map


# ── helper: build a synthetic user row in the training set ───────────────────
def _synthetic_user_id(train_df: pd.DataFrame) -> int:
    """Return an unused user_id (max + 1)."""
    return int(train_df["user_id"].max()) + 1


def _add_synthetic_user(train_df: pd.DataFrame, picks: list[dict]) -> pd.DataFrame:
    """Append synthetic user rows to a copy of train_df."""
    uid = _synthetic_user_id(train_df)
    rows = [{"user_id": uid, "anime_id": p["anime_id"], "rating": p["rating"]}
            for p in picks]
    return uid, pd.concat([train_df, pd.DataFrame(rows)], ignore_index=True)


# ── tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Data Analysis",
    "📈 Model Comparison",
    "🎌 Live Recommender",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Data Analysis")
    st.markdown(
        "Interactive EDA reports generated from the cleaned CooperUnion dataset "
        "(~12 K anime, ~7 M ratings)."
    )

    sub1, sub2 = st.tabs(["Anime", "Ratings"])

    with sub1:
        if os.path.exists(ANIME_HTML):
            with open(ANIME_HTML, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=900, scrolling=True)
        else:
            st.warning("anime_analysis.html not found — run `scripts/visualize_data.py` first.")

    with sub2:
        if os.path.exists(RATINGS_HTML):
            with open(RATINGS_HTML, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=900, scrolling=True)
        else:
            st.warning("ratings_analysis.html not found — run `scripts/visualize_data.py` first.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Comparison
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Model Comparison")
    st.markdown("Metrics evaluated on a 1 000-user random sample, Hit Rate / Precision / Recall @10.")

    tracker = load_tracker()

    # ── metrics table ────────────────────────────────────────────────────────
    st.subheader("Results table")
    metric_cols = ["Model", "Hit Rate @10", "Precision @10", "Recall @10", "Comments"]
    available   = [c for c in metric_cols if c in tracker.columns]
    st.dataframe(
        tracker[available].style.format(
            {c: "{:.4f}" for c in ["Hit Rate @10", "Precision @10", "Recall @10"]
             if c in tracker.columns}
        ).highlight_max(
            subset=["Hit Rate @10", "Precision @10", "Recall @10"],
            color="#d4edda",
        ),
        width="stretch",
        hide_index=True,
    )

    # ── bar charts ───────────────────────────────────────────────────────────
    st.subheader("Visual comparison")
    chart_metrics = [c for c in ["Hit Rate @10", "Precision @10", "Recall @10"]
                     if c in tracker.columns]

    cols = st.columns(len(chart_metrics))
    for col, metric in zip(cols, chart_metrics):
        with col:
            chart_data = tracker[["Model", metric]].set_index("Model")
            st.bar_chart(chart_data, width="stretch")
            st.caption(metric)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Live Recommender
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Live Recommender")
    st.markdown(
        "Pick **3 anime** you have watched and rate each one (1–10). "
        "All five models will recommend 10 anime for you side by side."
    )

    anime_df = load_anime()
    anime_names = sorted(anime_df["name"].dropna().unique().tolist())
    name_to_id  = dict(zip(anime_df["name"], anime_df["anime_id"]))

    # pick 3 stable random defaults (seeded so they don't change on every rerender)
    rng = np.random.default_rng(seed=7)
    default_names = rng.choice(anime_names, size=3, replace=False).tolist()

    def _filtered(q: str) -> list:
        """Exact → prefix → contains, case-insensitive. Falls back to full list."""
        q = q.strip().lower()
        if not q:
            return anime_names
        exact    = [n for n in anime_names if n.lower() == q]
        prefix   = [n for n in anime_names if n.lower().startswith(q) and n not in exact]
        contains = [n for n in anime_names if q in n.lower()
                    and n not in exact and n not in prefix]
        return (exact + prefix + contains) or anime_names

    # ── inject CSS to visually merge the text input and selectbox ────────────
    # ── selection (outside any form so search rerenders live) ────────────────
    st.subheader("Your 3 anime picks")
    picks = []
    for i in range(1, 4):
        st.markdown(f"**Anime #{i}**")
        col_search, col_match, col_rating = st.columns([2, 2, 1])
        with col_search:
            query = st.text_input(
                "Search",
                value="",
                placeholder=default_names[i - 1],
                key=f"search_{i}",
            )
        with col_match:
            options = _filtered(query)
            name = st.selectbox(
                "Best match",
                options=options,
                key=f"pick_{i}",
            )
        with col_rating:
            rating = st.slider(
                "Rating",
                min_value=1, max_value=10, value=7,
                key=f"rating_{i}",
            )
        resolved = name if name else default_names[i - 1]
        picks.append({"name": resolved, "anime_id": name_to_id.get(resolved), "rating": rating})

    with st.form("picker_form"):
        submitted = st.form_submit_button("🔍 Get Recommendations", width="stretch")

    if submitted:
        # validate — no duplicate picks, no missing IDs
        ids_chosen = [p["anime_id"] for p in picks if p["anime_id"] is not None]
        if len(set(ids_chosen)) < 3:
            st.error("Please pick 3 **different** anime.")
            st.stop()

        train_df = load_train()
        uid, augmented_train = _add_synthetic_user(train_df, picks)

        st.markdown("---")
        st.subheader(f"Top-10 recommendations for your profile")

        # ── display helper ────────────────────────────────────────────────────
        def show_recs(df: pd.DataFrame, score_col: str):
            """Merge with anime names and display a clean table."""
            if df.empty:
                st.warning("No recommendations returned.")
                return
            merged = df.merge(
                anime_df[["anime_id", "name", "genre"]],
                on="anime_id", how="left"
            )
            # guard: score_col or name/genre might be missing after a failed merge
            for col in ["name", "genre", score_col]:
                if col not in merged.columns:
                    st.warning(f"No recommendations returned (missing column: {col}).")
                    return
            display = merged[["name", "genre", score_col]].rename(
                columns={score_col: "score", "name": "Title", "genre": "Genres"}
            )
            display["score"] = display["score"].round(4)
            st.dataframe(display, width="stretch", hide_index=True)

        # ── run all 5 models ─────────────────────────────────────────────────
        model_cols = st.columns(5)

        # 1. Baseline
        with model_cols[0]:
            st.markdown("**Baseline**")
            with st.spinner("Loading…"):
                from src.models.baseline import compute_bayesian_scores, recommend_popular_anime
                scores_df = compute_bayesian_scores(augmented_train)
                recs = recommend_popular_anime(scores_df, augmented_train, user_id=uid, n=10)
                show_recs(recs, "bayesian_score")

        # 2. BoW
        with model_cols[1]:
            st.markdown("**BoW**")
            with st.spinner("Loading…"):
                from src.models.bow import recommend_bow
                bow_matrix = get_bow_matrix()
                recs = recommend_bow(uid, augmented_train, bow_matrix, anime_df, n=10)
                show_recs(recs, "bow_score")

        # 3. TF-IDF
        with model_cols[2]:
            st.markdown("**TF-IDF**")
            with st.spinner("Loading…"):
                from src.models.tfidf import recommend_tfidf
                tfidf_matrix = get_tfidf_matrix()
                recs = recommend_tfidf(uid, augmented_train, tfidf_matrix, anime_df, n=10)
                show_recs(recs, "tfidf_score")

        # 4. SVD
        with model_cols[3]:
            st.markdown("**SVD**")
            with st.spinner("Loading…"):
                from src.models.svd import build_user_item_matrix, train_svd, recommend_svd
                # rebuild user-item matrix including the synthetic user, re-run SVD
                matrix = build_user_item_matrix(augmented_train)
                recon  = train_svd(matrix, n_components=50)
                recs   = recommend_svd(uid, recon, augmented_train, n=10)
                show_recs(recs, "predicted_rating")

        # 5. NCF  ── NCF can only recommend for users it was trained on;
        #            since uid is new we fall back to the pre-trained model
        #            over the closest existing user (by rating overlap), or
        #            simply show a graceful fallback message.
        with model_cols[4]:
            st.markdown("**NCF**")
            with st.spinner("Loading…"):
                from src.models.ncf import recommend_ncf
                model_ncf, user_map, anime_map = get_ncf()
                if uid in user_map:
                    recs = recommend_ncf(uid, model_ncf, user_map, anime_map, augmented_train, n=10)
                    show_recs(recs, "predicted_rating")
                else:
                    st.info(
                        "NCF is trained on existing users only. "
                        "A new user profile isn't in the embedding table, "
                        "so NCF can't personalise here."
                    )
