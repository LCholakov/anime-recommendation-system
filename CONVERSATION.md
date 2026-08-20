# Conversation Log

## Session 1

---

### Bob explores the project

Before any requirements were given, Bob explored the full project structure:

- `src/`, `app/`, `tests/` — all empty, to be filled
- `data/anime.csv` — columns: `anime_id, name, genre, type, episodes, rating, members` (~12K titles)
- `data/rating.csv` — columns: `user_id, anime_id, rating` (rating `-1` = watched but not rated)
- `report/model_performance_tracker.xlsx` — empty Excel sheet
- `notes.md` — full TODO list + assignment requirements + sources (in Bulgarian)
- `requirements.txt` — empty

---

### User: Do you understand this project structure before I give you the details?

**Bob's summary:**

- **Project:** Anime Recommendation System — final project for a Deep Learning university course
- **Dataset:** CooperUnion anime-recommendations-database (Kaggle), smallest of three options considered, ~12K titles
- **Directories:** `src/` (core code), `app/` (Streamlit UI), `tests/` (BDD tests), `report/` (model Excel tracker)
- **Two mandatory deliverables:**
  1. BDD-style tests (`test_when_<condition>_then_<expectation>`, `unittest`, one test class per method)
  2. Model Report Excel (comparison table: hyperparams + metrics vs baseline, comments, train/val curves)

---

### User: I want to build a very simple anime recommendation system. I'm from linguistics, not STEM. What do you think of the plan in notes.md?

**Bob's assessment:**

**Good:**
- Order is logical: EDA → baseline → simple models → neural models → UI
- Maps 1:1 to assignment requirements
- Dataset choice (smallest option) is the right call for simplicity

**One concern — Step 8 (RNN):**
- The assignment says *"for example an RNN and an autoencoder"* — it's a suggestion, not a hard requirement
- An RNN works on sequences of user interactions **over time**, but `rating.csv` has **no timestamps**
- The data is not sequential → RNN would be forced and artificial on this dataset
- **Recommended replacement:** MLP (Multi-Layer Perceptron) — takes genre/feature embeddings, predicts rating, fits the data naturally, still counts as a neural network, papers cited even mention MLP as a deep learning baseline

**Assignment coverage:**

| Assignment step | Plan step |
|---|---|
| 1. Research papers | ✅ Done |
| 2. EDA | Step 3 |
| 3. TF-IDF + cosine vs BoW | Steps 5 & 6 |
| 4. SVD collaborative filtering | Step 7 |
| 5. Neural networks | Steps 8 & 9 |
| 6. Streamlit UI + BDD tests | Step 10 |
| 7. Presentation | Step 11 |

**Bottom line:** Plan is good. Swap RNN → MLP and you save significant effort for something that doesn't naturally fit the data.

---

### User: Adjust notes.md (keep old text, strikethrough it)

**Change made to `notes.md` step 8:**

```
~~RNN. Сравни с 5. И 7.~~ → MLP (Multi-Layer Perceptron). Сравни с 5. и 7. +тестове
   - Приема жанрови/характеристични ембединги и предсказва рейтинг.
   - По-подходящ от RNN — данните нямат времева последователност (няма timestamps в rating.csv).
```

---

---

### User: Will you remember this if I close the IDE?

**Bob:** No. Each new conversation starts with a blank context — no memory between sessions.

`CONVERSATION.md` is the workaround: paste its contents into a new chat and say "read this and continue where we left off."

---

### User: You're not automatically adding new conversation items to the conversation log? I need to instruct you every time?

**Bob:** No — should have been doing it from the start without being asked. Will append to `CONVERSATION.md` automatically at the end of every response going forward.

---
