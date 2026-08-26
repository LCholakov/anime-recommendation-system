import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

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


def train_ncf(
    train_df: pd.DataFrame,
    embed_dim: int = 32,
    epochs: int = 20,
    batch_size: int = 256,
    val_split: float = 0.2,
    patience: int = 3,
) -> tuple:
    user_map, anime_map = encode_ids(train_df)

    users   = torch.tensor(train_df["user_id"].map(user_map).values, dtype=torch.long)
    anime   = torch.tensor(train_df["anime_id"].map(anime_map).values, dtype=torch.long)
    ratings = torch.tensor(train_df["rating"].values, dtype=torch.float32)

    dataset    = TensorDataset(users, anime, ratings)
    val_size   = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model     = NCF(len(user_map), len(anime_map), embed_dim)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn   = nn.MSELoss()

    best_val_loss    = float("inf")
    patience_counter = 0
    best_state       = None

    for epoch in range(epochs):
        model.train()
        for u_batch, a_batch, r_batch in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(u_batch, a_batch), r_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_losses = [
                loss_fn(model(u, a), r).item()
                for u, a, r in val_loader
            ]
        val_loss = sum(val_losses) / len(val_losses)
        print(f"  Epoch {epoch+1:2d}/{epochs} — val_loss: {val_loss:.4f}")

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
