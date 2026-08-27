import pandas as pd
import numpy as np
import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED)


def encode_ids(train_df: pd.DataFrame) -> tuple:
    user_map  = {uid: i for i, uid in enumerate(sorted(train_df["user_id"].unique()))}
    anime_map = {aid: i for i, aid in enumerate(sorted(train_df["anime_id"].unique()))}
    return user_map, anime_map


class NCF(nn.Module):
    def __init__(self, n_users: int, n_anime: int, embed_dim: int = 32):
        super().__init__()
        self.user_emb  = nn.Embedding(n_users, embed_dim)
        self.anime_emb = nn.Embedding(n_anime, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, user_idx: torch.Tensor, anime_idx: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_idx)
        a = self.anime_emb(anime_idx)
        x = torch.cat([u, a], dim=1)
        return self.mlp(x).squeeze(1)


def _build_positive_sets(train_df: pd.DataFrame, user_map: dict, anime_map: dict) -> list:
    """Return a list indexed by encoded user id of the set of encoded positive anime ids."""
    n_users    = len(user_map)
    pos_sets   = [set() for _ in range(n_users)]
    for uid, aid in zip(train_df["user_id"], train_df["anime_id"]):
        u_idx = user_map.get(uid)
        a_idx = anime_map.get(aid)
        if u_idx is not None and a_idx is not None:
            pos_sets[u_idx].add(a_idx)
    return pos_sets


def _sample_bpr_batches(
    pos_sets: list,
    n_anime: int,
    n_per_user: int,
    rng: np.random.Generator,
) -> tuple:
    """Sample (user_idx, pos_anime_idx, neg_anime_idx) triples for BPR.

    For each user, draw n_per_user positives and match each with one random
    negative (an anime the user has not rated).
    """
    users_out = []
    pos_out   = []
    neg_out   = []

    for u_idx, pos in enumerate(pos_sets):
        if not pos:
            continue
        pos_list = list(pos)
        k        = min(n_per_user, len(pos_list))
        chosen   = rng.choice(pos_list, size=k, replace=False)
        for p in chosen:
            # sample a negative not in the positive set
            while True:
                n = int(rng.integers(0, n_anime))
                if n not in pos:
                    break
            users_out.append(u_idx)
            pos_out.append(p)
            neg_out.append(n)

    return (
        np.array(users_out, dtype=np.int64),
        np.array(pos_out,   dtype=np.int64),
        np.array(neg_out,   dtype=np.int64),
    )


def train_ncf(
    train_df: pd.DataFrame,
    embed_dim: int = 32,
    epochs: int = 20,
    batch_size: int = 256,
    val_split: float = 0.2,
    patience: int = 3,
    n_per_user: int = 10,
) -> tuple:
    """Train NCF with BPR (Bayesian Personalised Ranking) loss.

    Instead of predicting exact ratings (MSE), the model learns to rank:
    for each (user, pos_item, neg_item) triple drawn from the training data,
    the loss trains the model so that score(user, pos) > score(user, neg).
    This directly optimises top-N recommendation quality.

    Args:
        n_per_user: positive pairs to draw per user per epoch.
                    Higher → more training signal per epoch, slower per epoch.
    """
    rng = np.random.default_rng(SEED)
    user_map, anime_map = encode_ids(train_df)
    n_anime  = len(anime_map)

    pos_sets = _build_positive_sets(train_df, user_map, anime_map)

    model     = NCF(len(user_map), n_anime, embed_dim)
    optimizer = torch.optim.Adam(model.parameters())
    bce_loss  = nn.BCEWithLogitsLoss()

    best_val_loss    = float("inf")
    patience_counter = 0
    best_state       = None

    for epoch in range(epochs):
        u_arr, p_arr, n_arr = _sample_bpr_batches(pos_sets, n_anime, n_per_user, rng)
        total_pairs = len(u_arr)

        # split into train / val
        perm       = rng.permutation(total_pairs)
        val_size   = max(1, int(total_pairs * val_split))
        tr_idx     = perm[val_size:]
        vl_idx     = perm[:val_size]

        # ── training pass ────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for start in range(0, len(tr_idx), batch_size):
            batch   = tr_idx[start: start + batch_size]
            u_b     = torch.tensor(u_arr[batch], dtype=torch.long)
            p_b     = torch.tensor(p_arr[batch], dtype=torch.long)
            n_b     = torch.tensor(n_arr[batch], dtype=torch.long)

            pos_scores = model(u_b, p_b)
            neg_scores = model(u_b, n_b)

            # BPR via BCE: treat the pairwise diff as a logit
            diff   = pos_scores - neg_scores
            labels = torch.ones_like(diff)
            loss   = bce_loss(diff, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(u_b)

        train_loss /= max(len(tr_idx), 1)

        # ── validation pass ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for start in range(0, len(vl_idx), batch_size):
                batch   = vl_idx[start: start + batch_size]
                u_b     = torch.tensor(u_arr[batch], dtype=torch.long)
                p_b     = torch.tensor(p_arr[batch], dtype=torch.long)
                n_b     = torch.tensor(n_arr[batch], dtype=torch.long)
                diff    = model(u_b, p_b) - model(u_b, n_b)
                val_loss += bce_loss(diff, torch.ones_like(diff)).item() * len(u_b)
        val_loss /= max(len(vl_idx), 1)

        print(f"  Epoch {epoch+1:2d}/{epochs} — train_bpr: {train_loss:.4f}  val_bpr: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, user_map, anime_map


def find_proxy_user(
    picked_ids: list,
    user_map: dict,
    train_df: pd.DataFrame,
) -> int:
    """Return the existing user_id with the most overlap with picked_ids."""
    picked = set(picked_ids)
    user_anime = train_df[train_df["anime_id"].isin(picked)].groupby("user_id")["anime_id"].apply(set)
    if user_anime.empty:
        # fall back to any known user
        return next(iter(user_map))
    overlap = user_anime.apply(lambda s: len(s & picked))
    return int(overlap.idxmax())


def recommend_ncf(
    user_id: int,
    model: NCF,
    user_map: dict,
    anime_map: dict,
    train_df: pd.DataFrame,
    n: int = 10,
    exclude_ids: set = None,
) -> pd.DataFrame:
    if user_id not in user_map:
        return pd.DataFrame(columns=["anime_id", "predicted_rating"])

    rated_ids     = set(train_df[train_df["user_id"] == user_id]["anime_id"])
    excluded      = rated_ids | (exclude_ids or set())
    candidate_ids = [aid for aid in anime_map if aid not in excluded]
    if not candidate_ids:
        return pd.DataFrame(columns=["anime_id", "predicted_rating"])

    user_idx  = torch.tensor([user_map[user_id]] * len(candidate_ids), dtype=torch.long)
    anime_idx = torch.tensor([anime_map[aid] for aid in candidate_ids], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        scores = model(user_idx, anime_idx).numpy()

    top = pd.DataFrame({"anime_id": candidate_ids, "predicted_rating": scores})
    return top.nlargest(n, "predicted_rating").reset_index(drop=True)
