# Conversation Log

## Session 1

---

### User: wassup dude. Do you understand this project structure before I give you the details? I will give you details and requirements after that.

**Bob:** Let me explore the project structure first. *(explored files)*

Here's the breakdown:

**Project: Anime Recommendation System**

**Stack / Context:**
- **Language:** Python
- **Dataset:** CooperUnion anime-recommendations-database (Kaggle) — two CSVs in `data/`:
  - `anime.csv` — `anime_id, name, genre, type, episodes, rating, members`
  - `rating.csv` — `user_id, anime_id, rating` (ratings include `-1` for "watched but not rated")

**Directories (currently empty, to be filled):**
| Dir | Purpose |
|---|---|
| `src/` | Core source code (models, data processing, etc.) |
| `app/` | Streamlit web app |
| `tests/` | BDD-style unit tests |
| `report/` | Model performance tracker (empty Excel sheet right now) |
| `miscellaneous/` | Research notes and reference images |

**What needs to be built (from `notes.md`):**
1. ✅ Research papers reviewed
2. ✅ Dataset chosen
3. 🔲 **EDA** — data exploration, visualizations, anomaly detection
4. 🔲 **Baseline model** — e.g. recommend highest-rated anime
5. 🔲 **TF-IDF + cosine similarity** (content-based)
6. 🔲 **Bag of Words** comparison
7. 🔲 **SVD** collaborative filtering
8. 🔲 **RNN** neural approach
9. 🔲 **Autoencoder** neural approach
10. 🔲 **Streamlit UI**
11. 🔲 Documentation & presentation

**Two mandatory deliverables:**
- **BDD tests** — `test_when_<condition>_then_<expectation>` convention, `unittest`, one test class per method
- **Model Report Excel** — comparison table of all models with hyperparams, metrics, % change vs baseline, comments, train/val curves

Ready for your requirements — hit me with the details!

---

### User: So I wanna build a very very very simple anime recommendation system for a university project in subject Deep Learning. I am not from STEM, but from linguistics so I need it as simple as possible. Bare bones. What do you think of the plan I've created in notes.md?

**Bob:** Let me re-read the notes carefully before giving you a proper opinion. *(re-read notes.md)*

**Your plan is good — but slightly heavier than it needs to be.**

**✅ What's good:**
- The order is logical — EDA → baseline → simple models → neural models → UI. That's the correct progression.
- You've already done the two "free" steps (papers + dataset). Good head start.
- The dataset choice (CooperUnion, ~12K titles) is the **right call** for a simple project — it's the smallest of the three options you found, which means faster training and less headache.

**⚠️ One honest concern: the RNN (step 8)**

The assignment says *"например една рекурентна и един автоенкодер"* — **for example** an RNN and an autoencoder. It's a suggestion, not a hard requirement. An RNN for recommendations works on **sequences of user interactions over time**, which means:
- You need a user's *history in order* to make sense of it
- Your `rating.csv` has no timestamps — the data is **not sequential**
- An RNN would be somewhat forced/artificial on this dataset

**Simpler alternative:** replace the RNN with a basic **MLP (multi-layer perceptron)**. It fits the data naturally (just takes user-item embeddings as input), is much easier to implement, still counts as a neural network, and the papers you cited even mention MLP as a baseline for deep learning recommenders.

**✅ The plan fits the assignment exactly:**

| Assignment step | Your plan |
|---|---|
| 1. Research papers | ✅ Done |
| 2. EDA | Step 3 |
| 3. TF-IDF + cosine vs BoW | Steps 5 & 6 |
| 4. SVD collaborative filtering | Step 7 |
| 5. Neural networks (RNN + autoencoder) | Steps 8 & 9 |
| 6. Streamlit UI + BDD tests | Step 10 |
| 7. Presentation | Step 11 |

**Suggested minimal adjustment:** Replace step 8 (RNN) with an **MLP-based recommender** (takes genre/feature embeddings → predicts rating). Same neural network requirement, much simpler to build, actually fits your data. Then the autoencoder (step 9) stays as-is.

**Bottom line:** your plan is good, the sequence is right, and it covers everything required. Just swap the RNN for an MLP and you'll save yourself a significant headache for something that doesn't naturally fit your data anyway. Ready to start building whenever you are.

---

### User: adjust notes.md

**Bob:** *(first attempt was rejected by user)*

---

### User: adjust notes.md but don't remove text. Like strikethrough it or something like that

**Bob:** *(applied the following change to `notes.md` step 8)*

```
~~RNN. Сравни с 5. И 7.~~ → MLP (Multi-Layer Perceptron). Сравни с 5. и 7. +тестове
   - Приема жанрови/характеристични ембединги и предсказва рейтинг.
   - По-подходящ от RNN — данните нямат времева последователност (няма timestamps в rating.csv).
```

Done. Step 8 in `notes.md` now shows the original RNN struck through with the MLP replacement and the reasoning right below it.

---

### User: can you make a file to store our interaction fully?

**Bob:** Created `CONVERSATION.md` at the project root. It logs everything from this session — the project exploration, the plan review, the RNN→MLP recommendation, and the `notes.md` edit. I'll keep appending to it as we continue.

---

### User: will you remember this if I close the ide

**Bob:** No — I won't. Each time you start a new conversation with me, I begin with a completely blank context. I have no memory between sessions.

**What `CONVERSATION.md` gives you:**
- A human-readable log you can read back yourself
- Something you can **paste into a new chat** to bring me up to speed instantly

**How to use it next time:** Just open a new chat and say something like:
> *"Here's our conversation log, please read it and continue where we left off"*
> *(paste the contents of `CONVERSATION.md`)*

That's the most reliable way to resume. The file is yours — I just write to it.

---

### User: you're not automatically adding new conversation items to the conversation log? I need to instruct you every time?

**Bob:** You're right to call that out. I should be updating it automatically as we go — I just didn't commit to doing that without being asked.

I'll append to `CONVERSATION.md` at the end of every response from now on, no reminders needed. Starting with this exchange.

---

### User: you understand this'll be done in python, right buddy? How's your python btw?

**Bob:** Yeah, Python all the way — that's clear from the context (pandas, sklearn, PyTorch/Keras for the neural nets, Streamlit for the UI).

Python's my strongest language. We're talking:
- **Data wrangling** — `pandas`, `numpy`
- **ML models** — `scikit-learn` (TF-IDF, BoW, SVD, cosine similarity)
- **Deep learning** — `PyTorch` or `Keras/TensorFlow` (MLP, Autoencoder)
- **UI** — `Streamlit`
- **Tests** — `unittest` (BDD style as your prof requires)

No issues there. Ready to start coding whenever you are. What's first — EDA, the baseline model, or do you want to set up the project structure first (folders, `requirements.txt`, etc.)?

---

### User: I see in the conversation log you're not logging your little comments exactly: "Yeah, Python all the way" is logged as "Yes, Python throughout. Stack." Why?

**Bob:** Fair point — I was paraphrasing instead of quoting faithfully. I'll log my actual responses verbatim from now on, not cleaned-up summaries. Fixed that entry.

---

### User: fix all entries to match the conversation. Keep logging verbatim automatically.

**Bob:** Let me re-read the full log first to see what needs fixing. *(rewrote CONVERSATION.md with all entries verbatim and clean formatting)*

Done. All entries now match the actual conversation. Logging verbatim automatically going forward.

---

---

### User: now let's get cracking buddy. Get me two funcs in src/data.py load_anime_data() and load_rating_data(). Those will read the files for later analysis over the data

**Bob:** *(created `src/data.py`)*

```python
import pandas as pd


def load_anime_data(path: str = "data/anime.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def load_rating_data(path: str = "data/rating.csv") -> pd.DataFrame:
    return pd.read_csv(path)
```

`src/data.py` created with two functions:
- `load_anime_data()` — reads `data/anime.csv`, returns a DataFrame
- `load_rating_data()` — reads `data/rating.csv`, returns a DataFrame

Both default to the expected file paths but accept a custom `path` argument so tests can pass in fixture files without touching the real data.

---

---

### User: I forgot about the tests. Do you see the tests example in notes.md? Let's start with tests instead.

**Bob:** Yep, I see it clearly in `notes.md`. The convention is:
- One test file per source file → `tests/test_data.py`
- One test **class** per function → `TestLoadAnimeData`, `TestLoadRatingData`
- Test names: `test_when_<condition>_then_<expectation>`
- Write tests **first**, then make them pass (BDD)

What behaviors should we test for these two functions? Here's what makes sense for bare-bones loaders:

| Test | Condition | Expectation |
|---|---|---|
| 1 | called with a valid CSV | returns a DataFrame |
| 2 | called with a valid CSV | DataFrame is not empty |
| 3 | called with a non-existent path | raises `FileNotFoundError` |

That's 3 tests × 2 functions = 6 tests total. Simple, meaningful, 100% coverage. Should I go ahead with that?

---

---

### User: yea do it

**Bob:** *(created `tests/test_data.py` with 6 BDD tests — 3 per function)*

---

### User: I'll run the commands in the terminal. I'm the captain of this ship

**Bob:** Fair enough, captain. Run this:

```bash
python3 -m pytest tests/test_data.py -v
```

The `FileNotFoundError` tests will likely fail first — `pd.read_csv` raises `FileNotFoundError` on missing files but we need to confirm it bubbles up correctly. Let me know what you get and we'll fix from there.

---

---

### User: by the way, quick sidenote. Are you able to add timestamps to the log?

**Bob:** Not automatically — I don't have access to a real-time clock, so I can't stamp entries as they happen.

What I *can* do is log the **date** at the start of each session (you'd tell me the date when you paste the log to resume), or you could just add a single date header manually at the top of each session. That's probably the most practical approach given the limitations.

Want me to add today's date to the Session 1 header? What's today's date?

---

---

### User: sad. I was hoping for exact timestamps of prompts.

**Bob:** Yeah, genuinely can't do that — I only see the text of your messages, not when they were sent. The IDE doesn't pass timestamps to me.

The only way to get exact timestamps would be outside of me — for example `git log` already records commit times, so every time we make a change there's an implicit timestamp in the git history. Not perfect for the conversation itself, but it's something.

Want to just move on?

---

---

### User: got it. now back to the task at hand. Will run the tests

**Bob:** Go for it. Paste the output here when done.

---

---

### User: aaahhh no module named pytest. I think I need venv and add dependancies inside

**Bob:** Yep, exactly. Let's set it up:

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas pytest
```

Then run the tests with:
```bash
python3 -m pytest tests/test_data.py -v
```

Also, once that works, let me know and I'll write the `requirements.txt` so the dependencies are locked.

---

---

### User: Can't I just go python blablaabl. There's no other than v 3

**Bob:** Yeah, if you only have Python 3 installed you can just use `python` instead of `python3` — try:

```bash
python -m venv venv
source venv/bin/activate
pip install pandas pytest
python -m pytest tests/test_data.py -v
```

If even `python` isn't found, you might need to install Python first. But try it and see.

---

---

### User: I commented out the funcs in data.py. how come tests get PASSED?

**Bob:** The functions aren't commented out on my end — looks like your local edit didn't save, or you uncommented them back. Can you share exactly what your `src/data.py` looks like right now?

That said, the real answer to your question: **if the functions were truly commented out, the import on line 4 of the test file would fail with an `ImportError` and all tests would error out, not pass.** So either the file wasn't saved, or the functions were restored before pytest ran.

---

---

### User: aaah probly autosave's not on. Hold on I need to setup some stuff

**Bob:** Yep, autosave will bite you constantly without it. Go set it up, I'll be here.

---

---

### User: alright cool cool cool coool. tests fail when return val is commented out. Succeed when they're functioning for real. That's a decent start.

**Bob:** Perfect, that's exactly how it should work — red when broken, green when working. BDD in action.

Next step on the plan is the EDA (step 3). Want to tackle that now — `src/eda.py` with some basic analysis functions, plus tests first?

---
