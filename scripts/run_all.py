"""Run the full pipeline end-to-end in one command.

    python scripts/run_all.py

Steps (in order):
    1. Clean raw data  → data/anime_clean.csv, data/rating_clean.csv
    2. Split data      → data/train.csv, data/test.csv
    3. Sample eval set → data/eval_users.csv  (1 000 users, seed 42)
    4. Visualise data  → data/anime_analysis.html, data/ratings_analysis.html
    5. Baseline model  → model run + tracker entry
    6. BoW model       → model run + tracker entry
    7. TF-IDF model    → model run + tracker entry
    8. SVD model       → model run + tracker entry
    9. Autoencoder     → model run + tracker entry
   10. NCF             → model run + tracker entry
   11. Inspection report → report/manual_inspection.txt
   12. Launch Streamlit UI (replaces this process — Ctrl+C to stop)
"""

import sys
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PYTHON = sys.executable

STEPS = [
    ("Clean data",              "scripts/clean_data.py"),
    ("Split data",              "scripts/split_data.py"),
    ("Visualise data",          "scripts/visualize_data.py"),
    ("Run Baseline model",      "scripts/run_baseline.py"),
    ("Run BoW model",           "scripts/run_bow.py"),
    ("Run TF-IDF model",        "scripts/run_tfidf.py"),
    ("Run SVD model",           "scripts/run_svd.py"),
    ("Run Autoencoder",         "scripts/run_autoencoder.py"),
    ("Run NCF",                 "scripts/run_ncf.py"),
    ("Generate inspection report", "scripts/generate_inspection_report.py"),
]


def _banner(text: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {text}")
    print(f"{bar}")


def _sample_eval_users() -> None:
    """Sample 1 000 users from test.csv and save to data/eval_users.csv."""
    import pandas as pd
    test = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
    users = test["user_id"].drop_duplicates().sample(
        min(1000, test["user_id"].nunique()), random_state=42
    )
    out = os.path.join(ROOT, "data", "eval_users.csv")
    users.to_frame().to_csv(out, index=False)
    print(f"✅ Saved eval sample ({len(users)} users) → data/eval_users.csv")


def run_step(label: str, script: str) -> None:
    _banner(label)
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, os.path.join(ROOT, script)],
        cwd=ROOT,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n❌  Step '{label}' failed (exit {result.returncode}). Aborting.")
        sys.exit(result.returncode)
    print(f"\n✅  Done in {elapsed:.1f}s")


if __name__ == "__main__":
    total_start = time.time()

    for label, script in STEPS:
        run_step(label, script)

        # after split_data, regenerate the eval-user sample from the new test set
        if script == "scripts/split_data.py":
            _banner("Sample eval users")
            t0 = time.time()
            _sample_eval_users()
            print(f"✅  Done in {time.time() - t0:.1f}s")

    total = time.time() - total_start
    _banner(f"All steps complete — total time: {total:.0f}s")

    # Launch Streamlit — replaces this process so Ctrl+C stops it cleanly.
    _banner("Launching Streamlit UI  (Ctrl+C to stop)")
    app = os.path.join(ROOT, "app", "streamlit_app.py")
    os.execv(PYTHON, [PYTHON, "-m", "streamlit", "run", app])
