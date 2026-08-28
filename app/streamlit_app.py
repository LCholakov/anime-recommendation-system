import sys
import os
import html as html_module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Аниме Препоръки",
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


_MODEL_ORDER = [
    "Baseline (Popularity)",
    "BoW + Cosine Similarity",
    "TF-IDF + Cosine Similarity",
    "SVD Collaborative Filtering",
    "Autoencoder",
    "NCF (Neural Collaborative Filtering)",
]

def _apply_model_order(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_order"] = df["Model"].map({m: i for i, m in enumerate(_MODEL_ORDER)})
    return df.sort_values("_order").drop(columns="_order").reset_index(drop=True)


@st.cache_data
def load_tracker() -> pd.DataFrame:
    """Return one row per model — the best (highest Hit Rate @10) run."""
    df = pd.read_excel(TRACKER_XLSX)
    metric = "Hit Rate @10"
    if metric in df.columns:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        # only compare runs with exactly 10 recommendations
        if "n_recommendations" in df.columns:
            df = df[pd.to_numeric(df["n_recommendations"], errors="coerce") == 10]
        # keep the best (highest Hit Rate) run per model
        idx = df.groupby("Model")[metric].idxmax()
        best = df.loc[idx].reset_index(drop=True)
    else:
        best = df.drop_duplicates(subset=["Model"], keep="first").reset_index(drop=True)
    return _apply_model_order(best)


@st.cache_data
def load_tracker_all() -> pd.DataFrame:
    """Return all n_recommendations=10 runs (unsorted within model)."""
    df = pd.read_excel(TRACKER_XLSX)
    metric = "Hit Rate @10"
    if metric in df.columns:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    if "n_recommendations" in df.columns:
        df = df[pd.to_numeric(df["n_recommendations"], errors="coerce") == 10]
    return df.reset_index(drop=True)


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
    model = train_autoencoder(matrix, epochs=20, batch_size=128, patience=3, alpha=5.0)
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
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Анализ на данните",
    "📈 Сравнение на модели",
    "🎌 Препоръки на живо",
    "⚗️ Експерименти",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Анализ на данните")
    st.markdown(
        "Кратки доклади, генерирани от почистения датасет "
        "(~12 хил. аниме, ~7 млн. оценки)."
    )

    sub1, sub2 = st.tabs(["Аниме", "Оценки"])

    with sub1:
        if os.path.exists(ANIME_HTML):
            with open(ANIME_HTML, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=900, scrolling=True)
        else:
            st.warning("anime_analysis.html не е намерен — изпълнете `scripts/visualize_data.py` първо.")

    with sub2:
        if os.path.exists(RATINGS_HTML):
            with open(RATINGS_HTML, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=900, scrolling=True)
        else:
            st.warning("ratings_analysis.html не е намерен — изпълнете `scripts/visualize_data.py` първо.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Comparison
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Сравнение на модели")
    st.markdown("Метрики, изчислени върху произволна извадка от 1 000 потребители — Hit Rate @10.")

    tracker     = load_tracker()
    tracker_all = load_tracker_all()

    # ── metrics table ────────────────────────────────────────────────────────
    st.subheader("Таблица с резултати")
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
    st.subheader("Визуално сравнение")

    # Short display labels so bars are readable
    _short = {
        "Baseline (Popularity)":              "Baseline",
        "BoW + Cosine Similarity":            "BoW",
        "TF-IDF + Cosine Similarity":         "TF-IDF",
        "SVD Collaborative Filtering":        "SVD",
        "Autoencoder":                        "AE",
        "NCF (Neural Collaborative Filtering)":"NCF",
    }

    # Build second-best and avg-top-3 per model, in canonical order
    metric = "Hit Rate @10"
    _second_best = {}
    _avg_top3    = {}
    for model_name in _MODEL_ORDER:
        runs = (
            tracker_all[tracker_all["Model"] == model_name][metric]
            .dropna()
            .sort_values(ascending=False)
            .reset_index(drop=True)
        )
        _second_best[model_name] = float(runs.iloc[1]) if len(runs) >= 2 else None
        top3 = runs.iloc[:3]
        _avg_top3[model_name]    = float(top3.mean()) if len(top3) >= 1 else None

    # labels in canonical order (only models present in tracker)
    labels = [_short.get(m, m) for m in tracker["Model"]]

    import matplotlib.pyplot as _plt

    # `tracker["Model"]` is already in canonical order from load_tracker()
    _chart_models = tracker["Model"].tolist()
    _best_map     = dict(zip(tracker["Model"], tracker[metric]))

    def _make_bar(ax, values_map, color, title):
        vals      = [values_map.get(m) for m in _chart_models]
        plot_vals = [v if v is not None else 0.0 for v in vals]
        bars = ax.bar(labels, plot_vals, color=color)
        y_max = max((v for v in plot_vals if v), default=0.01) * 1.35
        ax.set_ylim(0, y_max)
        ax.set_title(title, fontsize=8, pad=4)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for bar, val in zip(bars, vals):
            label_txt = f"{val:.3f}" if val is not None else "n/a"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.02,
                label_txt, ha="center", va="bottom", fontsize=6,
            )

    col1, col2, col3 = st.columns(3)

    with col1:
        fig, ax = _plt.subplots(figsize=(3, 3))
        _make_bar(ax, _best_map, "#4C78A8", "Best — Hit Rate @10")
        _plt.tight_layout()
        st.pyplot(fig, width="stretch")
        _plt.close(fig)

    with col2:
        fig, ax = _plt.subplots(figsize=(3, 3))
        _make_bar(ax, _second_best, "#72A0C1", "2nd Best — Hit Rate @10")
        _plt.tight_layout()
        st.pyplot(fig, width="stretch")
        _plt.close(fig)

    with col3:
        fig, ax = _plt.subplots(figsize=(3, 3))
        _make_bar(ax, _avg_top3, "#A8C8A0", "Avg Top-3 — Hit Rate @10")
        _plt.tight_layout()
        st.pyplot(fig, width="stretch")
        _plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Live Recommender
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Препоръки на живо")
    st.markdown(
        "Изберете **3 аниме**, които сте гледали, и ги оценете (1–10). "
        "Всички модели ще ви препоръчат по 10 аниме един до друг."
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
    st.subheader("Вашите 3 аниме")
    picks = []
    for i in range(1, 4):
        st.markdown(f"**Аниме #{i}**")
        col_search, col_match, col_rating = st.columns([2, 2, 1])
        with col_search:
            query = st.text_input(
                "Търсене",
                value="",
                placeholder=default_names[i - 1],
                key=f"search_{i}",
            )
        with col_match:
            options = _filtered(query)
            name = st.selectbox(
                "Най-добро съвпадение",
                options=options,
                key=f"pick_{i}",
            )
        with col_rating:
            rating = st.slider(
                "Оценка",
                min_value=1, max_value=10, value=7,
                key=f"rating_{i}",
            )
        resolved = name if name else default_names[i - 1]
        picks.append({"name": resolved, "anime_id": name_to_id.get(resolved), "rating": rating})

    with st.form("picker_form"):
        submitted = st.form_submit_button("🔍 Препоръчай ми", width="stretch")

    if submitted:
        # validate — no duplicate picks, no missing IDs
        ids_chosen = [p["anime_id"] for p in picks if p["anime_id"] is not None]
        if len(set(ids_chosen)) < 3:
            st.error("Моля, изберете 3 **различни** аниме.")
            st.stop()

        train_df = load_train()
        uid, augmented_train = _add_synthetic_user(train_df, picks)

        st.markdown("---")
        st.subheader("Топ-10 препоръки за вашия профил")

        # ── display helper ────────────────────────────────────────────────────
        def show_recs(df: pd.DataFrame, score_col: str):
            """Display recommendations — merges name/genre only if not already present."""
            if df.empty:
                st.warning("Няма върнати препоръки.")
                return
            # BoW/TF-IDF already include name+genre; SVD/baseline/AE do not
            if "name" not in df.columns:
                df = df.merge(
                    anime_df[["anime_id", "name", "genre"]],
                    on="anime_id", how="left"
                )
            if score_col not in df.columns:
                st.warning(f"Няма върнати препоръки (липсва колона: {score_col}).")
                return
            display = df[["name", "genre", score_col]].rename(
                columns={score_col: "score", "name": "Заглавие", "genre": "Жанрове"}
            )
            display["score"] = display["score"].round(4)
            st.dataframe(display, width="stretch", hide_index=True)

        # ── run all 6 models ─────────────────────────────────────────────────
        model_cols = st.columns(6)

        # 1. Baseline
        with model_cols[0]:
            st.markdown("**Baseline**")
            with st.spinner("Зареждане…"):
                from src.models.baseline import compute_bayesian_scores, recommend_popular_anime
                scores_df = compute_bayesian_scores(augmented_train)
                recs = recommend_popular_anime(scores_df, augmented_train, user_id=uid, n=10)
                show_recs(recs, "bayesian_score")

        # 2. BoW
        with model_cols[1]:
            st.markdown("**BoW**")
            with st.spinner("Зареждане…"):
                from src.models.bow import recommend_bow
                bow_matrix = get_bow_matrix()
                recs = recommend_bow(uid, augmented_train, bow_matrix, anime_df, n=10)
                show_recs(recs, "bow_score")

        # 3. TF-IDF
        with model_cols[2]:
            st.markdown("**TF-IDF**")
            with st.spinner("Зареждане…"):
                from src.models.tfidf import recommend_tfidf
                tfidf_matrix = get_tfidf_matrix()
                recs = recommend_tfidf(uid, augmented_train, tfidf_matrix, anime_df, n=10)
                show_recs(recs, "tfidf_score")

        # 4. SVD — fold new user into cached Vt (no retraining)
        with model_cols[3]:
            st.markdown("**SVD**")
            with st.spinner("Зареждане…"):
                from src.models.svd import fold_in_user
                _, Vt, anime_cols = get_svd()
                recs = fold_in_user(picks, Vt, anime_cols, n=10)
                show_recs(recs, "predicted_rating")

        # 5. Autoencoder — forward-pass new user vector through cached model
        with model_cols[4]:
            st.markdown("**Автоенкодер**")
            with st.spinner("Зареждане…"):
                from src.models.autoencoder import recommend_autoencoder
                ae_model, ae_matrix = get_autoencoder()
                # Build a one-row vector aligned to the cached ae_matrix columns.
                # The matrix was built with center=True: each row is mean-centred
                # and scaled by /10.  We must apply the same transform to the
                # synthetic user so the model receives in-distribution input.
                raw_row = pd.Series(0.0, index=ae_matrix.columns)
                for p in picks:
                    if p["anime_id"] in raw_row.index:
                        raw_row[p["anime_id"]] = float(p["rating"])
                obs_vals = raw_row[raw_row != 0]
                if len(obs_vals) > 0:
                    row_mean = obs_vals.mean()
                    centered = raw_row.copy()
                    centered[raw_row != 0] = raw_row[raw_row != 0] - row_mean
                    user_row = centered / 10.0
                else:
                    user_row = raw_row
                ae_matrix_aug = pd.concat([ae_matrix, pd.DataFrame([user_row], index=[uid])])
                recs = recommend_autoencoder(uid, ae_model, ae_matrix_aug, augmented_train, n=10)
                show_recs(recs, "predicted_score")

        # 6. NCF — proxy user approach for new profiles
        with model_cols[5]:
            st.markdown("**NCF**")
            with st.spinner("Зареждане…"):
                from src.models.ncf import recommend_ncf, find_proxy_user
                model_ncf, user_map, anime_map = get_ncf()
                picked_ids = [p["anime_id"] for p in picks if p["anime_id"] is not None]
                proxy_id   = find_proxy_user(picked_ids, user_map, train_df)
                recs = recommend_ncf(
                    proxy_id, model_ncf, user_map, anime_map, train_df,
                    n=10, exclude_ids=set(picked_ids),
                )
                st.caption(f"*прокси потребител: {proxy_id}*")
                show_recs(recs, "predicted_rating")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Experiments
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    import time as _time
    import pickle as _pickle

    st.header("⚗️ Експерименти с модели")
    st.markdown(
        "Редактирайте параметрите на всеки модел, стартирайте тренировка и "
        "резултатите автоматично се записват в **model_performance_tracker.xlsx**."
    )

    # ── shared helpers ───────────────────────────────────────────────────────
    train_df_exp = load_train()
    anime_df_exp = load_anime()
    EVAL_CSV     = os.path.join(ROOT, "data", "eval_users.csv")

    def _run_eval(recommend_fn, n=10):
        from src.models.evaluator import evaluate
        test_df      = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
        sample_users = pd.read_csv(EVAL_CSV)["user_id"].tolist()
        return evaluate(recommend_fn, test_df[test_df["user_id"].isin(sample_users)],
                        train_df_exp, n=n)

    def _log(xlsx, name, metrics, hparams, comment):
        from src.models.evaluator import log_run
        log_run(xlsx, name, metrics, hparams, comment)

    # tracker model-name mapping (must match what log_run receives)
    _TRACKER_NAMES = {
        "Baseline":    "Baseline (Popularity)",
        "BoW":         "BoW + Cosine Similarity",
        "TF-IDF":      "TF-IDF + Cosine Similarity",
        "SVD":         "SVD Collaborative Filtering",
        "Autoencoder": "Autoencoder",
        "NCF":         "NCF (Neural Collaborative Filtering)",
    }

    # columns shown in the history table, per model
    _MODEL_COLS = {
        "Baseline":    ["Run #", "Timestamp", "m_percentile", "n_recommendations",
                        "Hit Rate @10", "Precision @10", "Recall @10", "What changed / Comment"],
        "BoW":         ["Run #", "Timestamp", "min_rating_threshold", "n_recommendations",
                        "Hit Rate @10", "Precision @10", "Recall @10", "What changed / Comment"],
        "TF-IDF":      ["Run #", "Timestamp", "min_rating_threshold", "sublinear_tf", "n_recommendations",
                        "Hit Rate @10", "Precision @10", "Recall @10", "What changed / Comment"],
        "SVD":         ["Run #", "Timestamp", "n_components", "n_recommendations", "Train time (s)",
                        "Hit Rate @10", "Precision @10", "Recall @10", "What changed / Comment"],
        "Autoencoder": ["Run #", "Timestamp", "epochs", "batch_size", "patience",
                        "train_users", "alpha", "n_recommendations", "Train time (s)",
                        "Hit Rate @10", "Precision @10", "Recall @10", "What changed / Comment"],
        "NCF":         ["Run #", "Timestamp", "embed_dim", "epochs", "batch_size",
                        "patience", "n_per_user", "n_recommendations", "Train time (s)",
                        "Hit Rate @10", "Precision @10", "Recall @10", "What changed / Comment"],
    }

    def _show_result(key, model_label):
        """Re-render persisted last-run result + full history table for this model."""
        r = st.session_state.get(key)
        if r:
            st.success(r["msg"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Hit Rate @K",  r["metrics"]["hit_rate"])
            c2.metric("Precision @K", r["metrics"]["precision"])
            c3.metric("Recall @K",    r["metrics"]["recall"])
            if r.get("epochs_log"):
                st.code(r["epochs_log"])

        # ── history table from tracker ────────────────────────────────────
        st.subheader("История на изпълненията")
        try:
            tracker_df = pd.read_excel(TRACKER_XLSX)
            model_name = _TRACKER_NAMES.get(model_label, model_label)
            hist = tracker_df[tracker_df["Model"] == model_name].copy()
            if hist.empty:
                st.caption("Няма записани изпълнения за този модел.")
            else:
                show_cols = [c for c in _MODEL_COLS.get(model_label, [
                    "Run #", "Timestamp",
                    "Hit Rate @10", "Precision @10", "Recall @10", "What changed / Comment",
                ]) if c in hist.columns]
                metric_cols = ["Hit Rate @10", "Precision @10", "Recall @10"]
                int_cols    = ["Run #", "m_percentile", "n_recommendations",
                               "min_rating_threshold",
                               "epochs", "batch_size", "patience", "embed_dim",
                               "n_components", "train_users",
                               "n_per_user"]
                display = hist[show_cols].reset_index(drop=True)
                # replace bare None with pd.NA so downstream coercions treat them uniformly
                display = display.infer_objects(copy=False).fillna(value=pd.NA)
                # force metric columns to float (N/A strings → NaN)
                for col in metric_cols:
                    if col in display.columns:
                        display[col] = pd.to_numeric(display[col], errors="coerce")
                # force integer columns to nullable Int64 so they show as whole numbers
                for col in int_cols:
                    if col in display.columns:
                        display[col] = pd.to_numeric(display[col], errors="coerce").astype("Int64")
                # coerce remaining object columns to str so Arrow serialisation never fails
                for col in display.columns:
                    if display[col].dtype == object:
                        display[col] = display[col].fillna("").astype(str)
                # only format columns that are actually float
                fmt = {c: "{:.4f}" for c in metric_cols if c in display.columns
                       and pd.api.types.is_float_dtype(display[c])}
                styler = display.style.format(fmt, na_rep="—")
                # faint green→yellow gradient on Hit Rate @10 — no matplotlib needed
                if "Hit Rate @10" in display.columns and display["Hit Rate @10"].notna().any():
                    hr_vals = pd.to_numeric(display["Hit Rate @10"], errors="coerce")
                    lo, hi  = hr_vals.min(), hr_vals.max()
                    spread  = hi - lo if hi != lo else 0.001
                    # small pad so equal values get a neutral mid-tone, not solid colour
                    pad = spread * 0.1

                    def _hr_colour(val):
                        try:
                            t = max(0.0, min(1.0, (float(val) - lo + pad) / (spread + 2 * pad)))
                        except (TypeError, ValueError):
                            return ""
                        # interpolate pale orange (255,235,210) → pale yellow (255,255,210) → pale green (215,255,210)
                        if t < 0.5:
                            s = t * 2
                            r = 255
                            g = int(235 + (255 - 235) * s)
                            b = 210
                        else:
                            s = (t - 0.5) * 2
                            r = int(255 + (215 - 255) * s)
                            g = 255
                            b = 210
                        return f"background-color: rgb({r},{g},{b})"

                    styler = styler.map(_hr_colour, subset=["Hit Rate @10"])
                st.dataframe(
                    styler,
                    hide_index=True,
                    width="stretch",
                )
        except Exception as _e:
            st.caption(f"Tracker не може да се зареди: {_e}")

    # ── model selector (radio keeps selection across reruns) ─────────────────
    EXP_MODELS = ["Baseline", "BoW", "TF-IDF", "SVD", "Autoencoder", "NCF"]
    selected = st.radio(
        "Модел",
        EXP_MODELS,
        horizontal=True,
        key="exp_model_select",
        label_visibility="collapsed",
    )
    st.divider()

    # ────────────────────────────────────────────────────────────────────────
    # Baseline
    # ────────────────────────────────────────────────────────────────────────
    if selected == "Baseline":
        st.subheader("Baseline — Bayesian Popularity")
        st.markdown("Препоръчва аниме с най-висок байесов рейтинг. Няма тренировка — само статистика върху train set.")

        with st.form("baseline_form"):
            c1, c2 = st.columns(2)
            with c1:
                bl_m_pct = st.slider(
                    "Праг m (перцентил на брой оценки)",
                    min_value=50, max_value=99, value=80,
                    help="Байесовият m = v.quantile(m_pct/100). По-висок → консервативни оценки."
                )
            with c2:
                bl_n = st.number_input("Препоръки @K", min_value=1, max_value=50, value=10)
            bl_comment = st.text_input("Коментар", value="Baseline popularity run")
            bl_run = st.form_submit_button("▶ Стартирай", type="primary")

        if bl_run:
            from src.models.baseline import recommend_popular_anime
            status = st.empty()
            t0 = _time.time()
            status.info("Изчисляване на байесови резултати…")
            scores_df = train_df_exp.groupby("anime_id")["rating"].agg(v="count", R="mean").reset_index()
            C = train_df_exp["rating"].mean()
            m = scores_df["v"].quantile(bl_m_pct / 100)
            scores_df["bayesian_score"] = (
                (scores_df["v"] / (scores_df["v"] + m)) * scores_df["R"] +
                (m / (scores_df["v"] + m)) * C
            )
            scores_bl = scores_df[["anime_id", "v", "R", "bayesian_score"]].rename(
                columns={"v": "rating_count", "R": "avg_rating"}
            )
            status.info("Оценяване на 1000 потребители…")
            metrics = _run_eval(
                lambda uid, n: recommend_popular_anime(scores_bl, train_df_exp, user_id=uid, n=n),
                n=int(bl_n)
            )
            elapsed = round(_time.time() - t0, 1)
            _log(TRACKER_XLSX, "Baseline (Popularity)", metrics,
                 {"split": "leave-one-out", "min_ratings": 5, "n_recommendations": int(bl_n),
                  "m_percentile": int(bl_m_pct)},
                 bl_comment)
            st.session_state["exp_result_baseline"] = {
                "msg": f"✅ Готово за {elapsed}s — записано в tracker",
                "metrics": metrics, "epochs_log": None,
            }
            status.empty()
            st.cache_data.clear()
            st.rerun()

        _show_result("exp_result_baseline", "Baseline")

    # ────────────────────────────────────────────────────────────────────────
    # BoW
    # ────────────────────────────────────────────────────────────────────────
    elif selected == "BoW":
        st.subheader("BoW + Cosine Similarity")
        st.markdown("Съдържателен модел — L2-нормализиран genre bag-of-words, dot-product cosine similarity.")

        with st.form("bow_form"):
            c1, c2 = st.columns(2)
            with c1:
                bow_n   = st.number_input("Препоръки @K", min_value=1, max_value=50, value=10)
                bow_mrt = st.number_input(
                    "min_rating_threshold", min_value=1, max_value=10, value=1,
                    help="Само оценки ≥ тази стойност се използват като семена за препоръки."
                )
            with c2:
                bow_comment = st.text_input("Коментар", value="BoW cosine run")
            bow_run = st.form_submit_button("▶ Стартирай", type="primary")

        if bow_run:
            from src.models.bow import build_bow_matrix, recommend_bow
            status = st.empty()
            t0 = _time.time()
            status.info("Изграждане на BoW матрица…")
            bow_m = build_bow_matrix(anime_df_exp)
            status.info(f"Матрица {bow_m.shape} — оценяване…")
            _bow_mrt = int(bow_mrt)
            metrics = _run_eval(
                lambda uid, n: recommend_bow(uid, train_df_exp, bow_m, anime_df_exp, n=n,
                                             min_rating_threshold=_bow_mrt),
                n=int(bow_n)
            )
            elapsed = round(_time.time() - t0, 1)
            _log(TRACKER_XLSX, "BoW + Cosine Similarity", metrics,
                 {"split": "leave-one-out", "min_ratings": 5, "n_recommendations": int(bow_n),
                  "min_rating_threshold": _bow_mrt},
                 bow_comment)
            st.session_state["exp_result_bow"] = {
                "msg": f"✅ Готово за {elapsed}s — записано в tracker",
                "metrics": metrics, "epochs_log": None,
            }
            status.empty()
            st.cache_data.clear()
            st.rerun()

        _show_result("exp_result_bow", "BoW")

    # ────────────────────────────────────────────────────────────────────────
    # TF-IDF
    # ────────────────────────────────────────────────────────────────────────
    elif selected == "TF-IDF":
        st.subheader("TF-IDF + Cosine Similarity")
        st.markdown("Рядко използваните жанрове получават по-голяма тежест от честите.")

        with st.form("tfidf_form"):
            c1, c2 = st.columns(2)
            with c1:
                tfidf_n   = st.number_input("Препоръки @K", min_value=1, max_value=50, value=10)
                tfidf_mrt = st.number_input(
                    "min_rating_threshold", min_value=1, max_value=10, value=1,
                    help="Само оценки ≥ тази стойност се използват като семена за препоръки."
                )
            with c2:
                tfidf_sublinear = st.checkbox(
                    "sublinear_tf",
                    value=False,
                    help="Замества честотата на жанра с 1+log(tf). Намалява влиянието на доминиращи жанрове."
                )
                tfidf_comment = st.text_input("Коментар", value="TF-IDF cosine run")
            tfidf_run = st.form_submit_button("▶ Стартирай", type="primary")

        if tfidf_run:
            from src.models.tfidf import build_tfidf_matrix, recommend_tfidf
            status = st.empty()
            t0 = _time.time()
            status.info("Изграждане на TF-IDF матрица…")
            tfidf_m = build_tfidf_matrix(anime_df_exp, sublinear_tf=bool(tfidf_sublinear))
            status.info(f"Матрица {tfidf_m.shape} — оценяване…")
            _tfidf_mrt = int(tfidf_mrt)
            metrics = _run_eval(
                lambda uid, n: recommend_tfidf(uid, train_df_exp, tfidf_m, anime_df_exp, n=n,
                                               min_rating_threshold=_tfidf_mrt),
                n=int(tfidf_n)
            )
            elapsed = round(_time.time() - t0, 1)
            _log(TRACKER_XLSX, "TF-IDF + Cosine Similarity", metrics,
                 {"split": "leave-one-out", "min_ratings": 5, "n_recommendations": int(tfidf_n),
                  "min_rating_threshold": _tfidf_mrt, "sublinear_tf": bool(tfidf_sublinear)},
                 tfidf_comment)
            st.session_state["exp_result_tfidf"] = {
                "msg": f"✅ Готово за {elapsed}s — записано в tracker",
                "metrics": metrics, "epochs_log": None,
            }
            status.empty()
            st.cache_data.clear()
            st.rerun()

        _show_result("exp_result_tfidf", "TF-IDF")

    # ────────────────────────────────────────────────────────────────────────
    # SVD
    # ────────────────────────────────────────────────────────────────────────
    elif selected == "SVD":
        st.subheader("SVD — Collaborative Filtering")
        st.markdown("Факторизира user-item матрицата с TruncatedSVD. Най-силният модел.")

        with st.form("svd_form"):
            c1, c2 = st.columns(2)
            with c1:
                svd_k = st.number_input(
                    "n_components (латентни фактори)",
                    min_value=5, max_value=300, value=50,
                    help="Брой сингулярни стойности. По-висок → повече детайл, по-бавно."
                )
            with c2:
                svd_n = st.number_input("Препоръки @K", min_value=1, max_value=50, value=10)
            svd_comment = st.text_input("Коментар", value="SVD run")
            svd_run = st.form_submit_button("▶ Стартирай", type="primary")

        if svd_run:
            from src.models.svd import build_user_item_matrix, train_svd, recommend_svd
            status = st.empty()
            t0 = _time.time()
            status.info("Изграждане на user-item матрица…")
            svd_matrix = build_user_item_matrix(train_df_exp)
            status.info(f"Матрица {svd_matrix.shape} — тренировка SVD (k={svd_k})…")
            t_train = _time.time()
            svd_recon, svd_Vt, svd_cols = train_svd(svd_matrix, n_components=int(svd_k))
            train_secs = round(_time.time() - t_train, 1)
            status.info(f"SVD готов ({train_secs}s) — оценяване…")
            metrics = _run_eval(
                lambda uid, n: recommend_svd(uid, svd_recon, train_df_exp, n=n),
                n=int(svd_n)
            )
            elapsed = round(_time.time() - t0, 1)
            with open(SVD_PKL, "wb") as _f:
                _pickle.dump((svd_recon, svd_Vt, svd_cols), _f)
            _log(TRACKER_XLSX, "SVD Collaborative Filtering", metrics,
                 {"split": "leave-one-out", "min_ratings": 5, "n_recommendations": int(svd_n),
                  "n_components": int(svd_k), "train_secs": train_secs},
                 svd_comment)
            st.session_state["exp_result_svd"] = {
                "msg": f"✅ Готово за {elapsed}s (тренировка: {train_secs}s) — записано в tracker",
                "metrics": metrics, "epochs_log": None,
            }
            status.empty()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        _show_result("exp_result_svd", "SVD")

    # ────────────────────────────────────────────────────────────────────────
    # Autoencoder
    # ────────────────────────────────────────────────────────────────────────
    elif selected == "Autoencoder":
        st.subheader("Autoencoder")
        st.markdown("Dense 128 → 32 → 128, Sigmoid, **weighted MSE** (implicit feedback, confidence weight α), mean-centred user rows. Тренира се върху подизвадка от потребители.")

        with st.form("ae_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                ae_epochs      = st.number_input("Епохи (макс)", min_value=1,   max_value=500,   value=200)
                ae_batch       = st.number_input("Batch size",   min_value=16,  max_value=512,   value=128, step=16)
            with c2:
                ae_patience    = st.number_input("Patience",     min_value=1,   max_value=30,    value=10)
                ae_train_users = st.number_input(
                    "Train потребители", min_value=500, max_value=60000, value=10000, step=500,
                    help="Подизвадка. По-голяма → по-добра матрица, по-бавно."
                )
            with c3:
                ae_alpha = st.number_input(
                    "alpha", min_value=0.1, max_value=20.0, value=5.0, step=0.5,
                    help="Тежест на наблюдаваните елементи в weighted MSE. По-голямо → по-силен акцент върху оценените аниме."
                )
                ae_n = st.number_input("Препоръки @K", min_value=1, max_value=50, value=10)
            ae_comment = st.text_input("Коментар", value="Autoencoder v3 run — target>0 mask, epochs=200, patience=10")
            ae_run = st.form_submit_button("▶ Стартирай", type="primary")

        if ae_run:
            import torch as _torch
            import builtins as _builtins
            from src.models.autoencoder import (
                build_user_item_matrix as ae_build_matrix,
                train_autoencoder, recommend_autoencoder,
            )
            status    = st.empty()
            epoch_box = st.empty()
            t0 = _time.time()

            status.info(f"Подизвадка {int(ae_train_users)} потребители…")
            sampled = pd.Series(train_df_exp["user_id"].unique()).sample(
                min(int(ae_train_users), train_df_exp["user_id"].nunique()), random_state=42
            ).tolist()
            train_sub = train_df_exp[train_df_exp["user_id"].isin(sampled)]
            status.info(f"Изграждане на матрица ({len(sampled)} потребители)…")
            ae_matrix = ae_build_matrix(train_sub)
            status.info(f"Матрица {ae_matrix.shape} — тренировка…")

            epoch_lines = []
            _orig_print = _builtins.print
            def _cap_print(*args, **kwargs):
                msg = " ".join(str(a) for a in args)
                if "Epoch" in msg or "Early" in msg:
                    epoch_lines.append(msg.strip())
                    epoch_box.code("\n".join(epoch_lines[-15:]))
                _orig_print(*args, **kwargs)
            _builtins.print = _cap_print
            t_train = _time.time()
            try:
                ae_model = train_autoencoder(
                    ae_matrix, epochs=int(ae_epochs),
                    batch_size=int(ae_batch), patience=int(ae_patience),
                    alpha=float(ae_alpha),
                )
            finally:
                _builtins.print = _orig_print

            train_secs = round(_time.time() - t_train, 1)
            status.info(f"Тренировка завършена ({train_secs}s) — оценяване…")
            metrics = _run_eval(
                lambda uid, n: recommend_autoencoder(uid, ae_model, ae_matrix, train_sub, n=n),
                n=int(ae_n)
            )
            elapsed = round(_time.time() - t0, 1)
            _torch.save(ae_model.state_dict(), AE_PT)
            with open(AE_MATRIX_PKL, "wb") as _f:
                _pickle.dump(ae_matrix, _f)
            _log(TRACKER_XLSX, "Autoencoder", metrics,
                 {"split": "leave-one-out", "min_ratings": 5, "n_recommendations": int(ae_n),
                  "epochs": int(ae_epochs), "batch_size": int(ae_batch),
                  "patience": int(ae_patience), "train_users": int(ae_train_users),
                  "alpha": float(ae_alpha), "train_secs": train_secs},
                 ae_comment)
            st.session_state["exp_result_ae"] = {
                "msg": f"✅ Готово за {elapsed}s (тренировка: {train_secs}s) — записано в tracker",
                "metrics": metrics,
                "epochs_log": "\n".join(epoch_lines),
            }
            status.empty()
            epoch_box.empty()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        _show_result("exp_result_ae", "Autoencoder")

    # ────────────────────────────────────────────────────────────────────────
    # NCF
    # ────────────────────────────────────────────────────────────────────────
    elif selected == "NCF":
        st.subheader("NCF — Neural Collaborative Filtering")
        st.markdown("User + anime embeddings → concat → Dense 64 → Dense 32 → 1. **BPR ranking loss** (positive/negative item pairs), Adam.")
        with st.form("ncf_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                ncf_embed    = st.number_input("Embedding dim", min_value=4,  max_value=256,  value=32,  step=4)
                ncf_epochs   = st.number_input("Епохи (макс)",  min_value=1,  max_value=100,  value=20)
            with c2:
                ncf_batch    = st.number_input("Batch size",    min_value=32, max_value=2048, value=256, step=32)
                ncf_patience = st.number_input("Patience",      min_value=1,  max_value=20,   value=3)
            with c3:
                ncf_n_per_user = st.number_input(
                    "n_per_user", min_value=1, max_value=50, value=10,
                    help="BPR двойки (позитивна/негативна) на потребител на епоха. По-голямо → по-силен сигнал, по-бавно."
                )
                ncf_n = st.number_input("Препоръки @K", min_value=1, max_value=50, value=10)
            ncf_comment = st.text_input("Коментар", value="NCF v2 run")
            ncf_run = st.form_submit_button("▶ Стартирай", type="primary")

        if ncf_run:
            import torch as _torch
            import builtins as _builtins
            from src.models.ncf import train_ncf, recommend_ncf
            status    = st.empty()
            epoch_box = st.empty()
            t0 = _time.time()
            status.info(f"Тренировка NCF (embed={ncf_embed}, epochs={ncf_epochs}, batch={ncf_batch})…")

            epoch_lines = []
            _orig_print = _builtins.print
            def _cap_print(*args, **kwargs):
                msg = " ".join(str(a) for a in args)
                if "Epoch" in msg or "Early" in msg:
                    epoch_lines.append(msg.strip())
                    epoch_box.code("\n".join(epoch_lines[-15:]))
                _orig_print(*args, **kwargs)
            _builtins.print = _cap_print
            t_train = _time.time()
            try:
                ncf_model, ncf_umap, ncf_amap = train_ncf(
                    train_df_exp, embed_dim=int(ncf_embed),
                    epochs=int(ncf_epochs), batch_size=int(ncf_batch),
                    patience=int(ncf_patience), n_per_user=int(ncf_n_per_user),
                )
            finally:
                _builtins.print = _orig_print

            train_secs = round(_time.time() - t_train, 1)
            status.info(f"Тренировка завършена ({train_secs}s) — оценяване…")
            metrics = _run_eval(
                lambda uid, n: recommend_ncf(uid, ncf_model, ncf_umap, ncf_amap, train_df_exp, n=n),
                n=int(ncf_n)
            )
            elapsed = round(_time.time() - t0, 1)
            _torch.save(ncf_model.state_dict(), NCF_PT)
            with open(NCF_MAPS_PKL, "wb") as _f:
                _pickle.dump((ncf_umap, ncf_amap), _f)
            _log(TRACKER_XLSX, "NCF (Neural Collaborative Filtering)", metrics,
                 {"split": "leave-one-out", "min_ratings": 5, "n_recommendations": int(ncf_n),
                  "epochs": int(ncf_epochs), "batch_size": int(ncf_batch),
                  "patience": int(ncf_patience), "embed_dim": int(ncf_embed),
                  "n_per_user": int(ncf_n_per_user), "train_secs": train_secs},
                 ncf_comment)
            st.session_state["exp_result_ncf"] = {
                "msg": f"✅ Готово за {elapsed}s (тренировка: {train_secs}s) — записано в tracker",
                "metrics": metrics,
                "epochs_log": "\n".join(epoch_lines),
            }
            status.empty()
            epoch_box.empty()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        _show_result("exp_result_ncf", "NCF")
