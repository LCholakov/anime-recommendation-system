# anime-recommendation-system
Final project for Introduction to deep learning university course. Will (attempt) to build a system that reccommends anime. 

Install requirements from `requirements.txt`.
Run everything with `python scripts/run_all.py`

-----------------------------------------

Script run_all basically does the following: 

Check if data files anime.csv and rating.csv exist in data dir.
If not, extract data/data.zip to place raw data files there.

Clean data with `python scripts/clean_data.py` 

Analyze data with `python scripts/analyze_data.py`

Vizualize creates a little html file `python scripts/visualize_data.py` 

run model scripts with 
baseline (popularity) `python scripts/run_baseline.py`
Bag of Words `python scripts/run_bow.py`
TF-IDF `python scripts/run_baseline.py`
Autoencoder `python scripts/run_baseline.py`
NCF `python scripts/run_baseline.py`

Start UI with
`streamlit run app/streamlit_app.py`

The above steps can be executed manually as well. 