# anime-recommendation-system
Final project for Introduction to deep learning university course. Will (attempt) to build a system that reccommends anime. 

Install requirements from `requirements.txt`.
Run everything with `python scripts/run_all.py`

-----------------------------------------

Script run_all basically does the following: 

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

The above steps can be executed manually as well. 