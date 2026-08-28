import pandas as pd
import numpy as np
import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED)


class AnimeAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def build_user_item_matrix(train_df: pd.DataFrame, center: bool = True) -> pd.DataFrame:
    """Build a normalised user-item rating matrix.

    Args:
        center: if True (default), subtract each user's mean observed rating
                from their observed items before filling zeros.  This makes the
                model learn relative preference (above/below average) rather
                than absolute scores, which dramatically improves performance on
                sparse matrices where zero-cells otherwise dominate the loss.
    """
    matrix = train_df.pivot_table(
        index="user_id", columns="anime_id", values="rating", aggfunc="mean"
    ).fillna(0)
    if center:
        # per-user mean of observed (non-zero) items
        observed_mask = matrix.values != 0
        row_means = np.where(
            observed_mask.sum(axis=1, keepdims=True) > 0,
            (matrix.values * observed_mask).sum(axis=1, keepdims=True)
            / observed_mask.sum(axis=1, keepdims=True).clip(min=1),
            0.0,
        )
        # subtract mean from observed cells only; zeros stay zero
        matrix = pd.DataFrame(
            np.where(observed_mask, matrix.values - row_means, 0.0),
            index=matrix.index,
            columns=matrix.columns,
        )
        # scale to [-0.5, 0.5] so Sigmoid mid-point (0.5) corresponds to
        # "average" and the model can output above/below symmetrically
        matrix = matrix / 10.0
    else:
        matrix = matrix / 10.0
    return matrix


def _weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Weighted MSE for implicit-feedback collaborative filtering.

    Above-average items (target > 0 after mean-centering) receive weight
    (1 + alpha); all other cells receive weight 1.  After mean-centering,
    target > 0 identifies items the user rated *above* their own average —
    exactly the positive signal we want to reinforce for top-N ranking.
    Below-average items and unobserved zeros are both treated as neutral,
    which matches the recommender objective: we care about surfacing things
    the user will like above average, not penalising items they merely
    rated mediocrely.

    Reference: Hu et al., "Collaborative Filtering for Implicit Feedback
    Datasets", ICDM 2008.
    """
    mask    = (target > 0).float()           # above-average observed items
    weights = 1.0 + alpha * mask             # (B, n_items)
    loss    = (weights * (pred - target) ** 2).mean()
    return loss


def train_autoencoder(
    matrix: pd.DataFrame,
    epochs: int = 200,
    batch_size: int = 128,
    val_split: float = 0.2,
    patience: int = 10,
    alpha: float = 5.0,
) -> "AnimeAutoencoder":
    """Train the autoencoder with weighted MSE (implicit feedback loss).

    Each forward pass reconstructs the full user rating vector.  Items the
    user rated *above* their own average (target > 0 after mean-centering)
    receive loss weight (1 + alpha) — the model is pushed to score those items
    higher than everything else.  Below-average rated items and unobserved
    zeros both receive weight 1, treating them as neutral.  This gives the
    correct top-N ranking signal: reinforce positives, don't fight neutrals.

    Dense gradients across all 9k items every step — correct for an
    autoencoder-based collaborative filter (unlike BPR which only touches
    2 positions per sample).

    Args:
        alpha: confidence weight for above-average observed items.
               Typical range 1–10.  Higher → stronger push to rank
               above-average items at the top.
    """
    torch.manual_seed(SEED)                                      # reproducible init
    rng       = np.random.default_rng(SEED)
    vals      = matrix.values.astype(np.float32)
    X         = torch.tensor(vals)                               # (n_users, n_items)
    n         = len(X)

    val_size   = max(1, int(n * val_split))
    train_size = n - val_size
    perm       = rng.permutation(n)
    tr_idx     = perm[:train_size]
    vl_idx     = perm[train_size:]

    model     = AnimeAutoencoder(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters())

    best_val_loss    = float("inf")
    patience_counter = 0
    best_state       = None

    for epoch in range(epochs):
        # ── training ────────────────────────────────────────────────────────
        model.train()
        rng.shuffle(tr_idx)
        train_loss = 0.0
        for start in range(0, len(tr_idx), batch_size):
            batch = tr_idx[start: start + batch_size]
            rows  = X[batch]
            optimizer.zero_grad()
            loss  = _weighted_mse(model(rows), rows, alpha)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        train_loss /= max(len(tr_idx), 1)

        # ── validation ──────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for start in range(0, len(vl_idx), batch_size):
                batch = vl_idx[start: start + batch_size]
                rows  = X[batch]
                val_loss += _weighted_mse(model(rows), rows, alpha).item() * len(batch)
        val_loss /= max(len(vl_idx), 1)

        print(f"  Epoch {epoch+1:2d}/{epochs} — train_wmse: {train_loss:.6f}  val_wmse: {val_loss:.6f}")

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
    return model


def recommend_autoencoder(
    user_id: int,
    model: "AnimeAutoencoder",
    matrix: pd.DataFrame,
    train_df: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    if user_id not in matrix.index:
        return pd.DataFrame(columns=["anime_id", "predicted_score"])

    rated_ids = set(train_df[train_df["user_id"] == user_id]["anime_id"])
    x = torch.tensor(matrix.loc[user_id].values, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        scores = model(x).squeeze(0).numpy()

    score_series = pd.Series(scores, index=matrix.columns)
    score_series = score_series.drop(index=[i for i in rated_ids if i in score_series.index])
    top = score_series.nlargest(n).reset_index()
    top.columns = ["anime_id", "predicted_score"]
    return top
