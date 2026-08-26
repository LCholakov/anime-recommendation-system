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

MODEL_DIR       = os.path.join(ROOT, "model")
BOW_PKL         = os.path.join(MODEL_DIR, "bow_matrix.pkl")
TFIDF_PKL       = os.path.join(MODEL_DIR, "tfidf_matrix.pkl")
SVD_PKL         = os.path.join(MODEL_DIR, "svd_reconstructed.pkl")
AE_PT           = os.path.join(MODEL_DIR, "autoencoder.pt")
AE_MATRIX_PKL   = os.path.join(MODEL_DIR, "ae_matrix.pkl")
NCF_PT          = os.path.join(MODEL_DIR, "ncf.pt")
NCF_MAPS_PKL    = os.path.join(MODEL_DIR, "ncf_maps.pkl")


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


# ── cached model builders (load from disk if saved, else build + save) ───────
import pickle
import torch

os.makedirs(MODEL_DIR, exist_ok=True)


@st.cache_resource
def get_bow_matrix():
    import pickle
    from src.models.bow import build_bow_matrix
    if os.path.exists(BOW_PKL):
        with open(BOW_PKL, "rb") as f:
            return pickle.load(f)
    matrix = build_bow_matrix(load_anime())
    with open(BOW_PKL, "wb") as f:
        pickle.dump(matrix, f)
    return matrix


@st.cache_resource
def get_tfidf_matrix():
    import pickle
    from src.models.tfidf import build_tfidf_matrix
    if os.path.exists(TFIDF_PKL):
        with open(TFIDF_PKL, "rb") as f:
            return pickle.load(f)
    matrix = build_tfidf_matrix(load_anime())
    with open(TFIDF_PKL, "wb") as f:
        pickle.dump(matrix, f)
    return matrix


@st.cache_resource
def get_svd():
    """Returns (reconstructed_df, Vt, anime_columns)."""
    import pickle
    from src.models.svd import build_user_item_matrix, train_svd
    if os.path.exists(SVD_PKL):
        with open(SVD_PKL, "rb") as f:
            return pickle.load(f)
    train = load_train()
    matrix = build_user_item_matrix(train)
    result = train_svd(matrix, n_components=50)  # returns (recon_df, Vt, cols)
    with open(SVD_PKL, "wb") as f:
        pickle.dump(result, f)
    return result


@st.cache_resource
def get_autoencoder():
    import pickle, torch
    from src.models.autoencoder import AnimeAutoencoder, build_user_item_matrix, train_autoencoder
    if os.path.exists(AE_PT) and os.path.exists(AE_MATRIX_PKL):
        with open(AE_MATRIX_PKL, "rb") as f:
            matrix = pickle.load(f)
        model = AnimeAutoencoder(matrix.shape[1])
        model.load_state_dict(torch.load(AE_PT, map_location="cpu", weights_only=True))
        model.eval()
        return model, matrix
    train = load_train()
    matrix = build_user_item_matrix(train)
    model = train_autoencoder(matrix, epochs=20, batch_size=128, patience=3)
    torch.save(model.state_dict(), AE_PT)
    with open(AE_MATRIX_PKL, "wb") as f:
        pickle.dump(matrix, f)
    return model, matrix


@st.cache_resource
def get_ncf():
    import pickle, torch
    from src.models.ncf import NCF, encode_ids, train_ncf
    if os.path.exists(NCF_PT) and os.path.exists(NCF_MAPS_PKL):
        with open(NCF_MAPS_PKL, "rb") as f:
            user_map, anime_map = pickle.load(f)
        model = NCF(len(user_map), len(anime_map), embed_dim=32)
        model.load_state_dict(torch.load(NCF_PT, map_location="cpu", weights_only=True))
        model.eval()
        return model, user_map, anime_map
    train = load_train()
    model, user_map, anime_map = train_ncf(
        train, embed_dim=32, epochs=20, batch_size=256, patience=3
    )
    torch.save(model.state_dict(), NCF_PT)
    with open(NCF_MAPS_PKL, "wb") as f:
        pickle.dump((user_map, anime_map), f)
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

    # ── tab-order: search1 → search2 → search3, selectboxes removed from tab order ──
    st.components.v1.html("""
    <script>
    function fixTabOrder() {
        const doc = window.parent.document;

        // Pull out all plain text inputs (the Search boxes)
        const searchInputs = Array.from(
            doc.querySelectorAll('input[type="text"]')
        ).slice(0, 3);   // first 3 are our search boxes

        // Remove selectbox inner inputs from tab order
        doc.querySelectorAll('div[data-baseweb="select"] input').forEach(el => {
            el.setAttribute("tabindex", "-1");
        });

        // Assign tabindex 1-3 to search boxes and autofocus the first
        searchInputs.forEach((el, i) => {
            el.setAttribute("tabindex", String(i + 1));
        });
        if (searchInputs.length > 0) searchInputs[0].focus();
    }

    setTimeout(fixTabOrder, 800);
    </script>
    """, height=0)

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
            """Display recommendations — merges name/genre only if not already present."""
            if df.empty:
                st.warning("No recommendations returned.")
                return
            # BoW/TF-IDF already include name+genre; SVD/baseline/AE do not
            if "name" not in df.columns:
                df = df.merge(
                    anime_df[["anime_id", "name", "genre"]],
                    on="anime_id", how="left"
                )
            if score_col not in df.columns:
                st.warning(f"No recommendations returned (missing score column: {score_col}).")
                return
            display = df[["name", "genre", score_col]].rename(
                columns={score_col: "score", "name": "Title", "genre": "Genres"}
            )
            display["score"] = display["score"].round(4)
            st.dataframe(display, width="stretch", hide_index=True)

        # ── run all 6 models ─────────────────────────────────────────────────
        model_cols = st.columns(6)

        # 1. Baseline
        with model_cols[0]:
            st.markdown("**Baseline**")
            with st.spinner("Running…"):
                from src.models.baseline import compute_bayesian_scores, recommend_popular_anime
                scores_df = compute_bayesian_scores(augmented_train)
                recs = recommend_popular_anime(scores_df, augmented_train, user_id=uid, n=10)
                show_recs(recs, "bayesian_score")

        # 2. BoW
        with model_cols[1]:
            st.markdown("**BoW**")
            with st.spinner("Running…"):
                from src.models.bow import recommend_bow
                bow_matrix = get_bow_matrix()
                recs = recommend_bow(uid, augmented_train, bow_matrix, anime_df, n=10)
                show_recs(recs, "bow_score")

        # 3. TF-IDF
        with model_cols[2]:
            st.markdown("**TF-IDF**")
            with st.spinner("Running…"):
                from src.models.tfidf import recommend_tfidf
                tfidf_matrix = get_tfidf_matrix()
                recs = recommend_tfidf(uid, augmented_train, tfidf_matrix, anime_df, n=10)
                show_recs(recs, "tfidf_score")

        # 4. SVD — fold new user into cached Vt (no retraining)
        with model_cols[3]:
            st.markdown("**SVD**")
            with st.spinner("Running…"):
                from src.models.svd import fold_in_user
                _, Vt, anime_cols = get_svd()
                recs = fold_in_user(picks, Vt, anime_cols, n=10)
                show_recs(recs, "predicted_rating")

        # 5. Autoencoder — forward-pass new user vector through cached model
        with model_cols[4]:
            st.markdown("**Autoencoder**")
            with st.spinner("Running…"):
                from src.models.autoencoder import recommend_autoencoder
                ae_model, ae_matrix = get_autoencoder()
                # build a one-row sparse vector aligned to the cached ae_matrix columns
                user_row = pd.Series(0.0, index=ae_matrix.columns)
                for p in picks:
                    if p["anime_id"] in user_row.index:
                        user_row[p["anime_id"]] = p["rating"] / 10.0
                ae_matrix_aug = pd.concat([ae_matrix, pd.DataFrame([user_row], index=[uid])])
                recs = recommend_autoencoder(uid, ae_model, ae_matrix_aug, augmented_train, n=10)
                show_recs(recs, "predicted_score")

        # 6. NCF — proxy user approach for new profiles
        with model_cols[5]:
            st.markdown("**NCF**")
            with st.spinner("Running…"):
                from src.models.ncf import recommend_ncf, find_proxy_user
                model_ncf, user_map, anime_map = get_ncf()
                picked_ids = [p["anime_id"] for p in picks if p["anime_id"] is not None]
                proxy_id   = find_proxy_user(picked_ids, user_map, train_df)
                recs = recommend_ncf(
                    proxy_id, model_ncf, user_map, anime_map, train_df,
                    n=10, exclude_ids=set(picked_ids),
                )
                st.caption(f"*proxy user: {proxy_id}*")
                show_recs(recs, "predicted_rating")
