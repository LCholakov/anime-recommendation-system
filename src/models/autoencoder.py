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


def build_user_item_matrix(train_df: pd.DataFrame) -> pd.DataFrame:
    matrix = train_df.pivot_table(
        index="user_id", columns="anime_id", values="rating", aggfunc="mean"
    ).fillna(0)
    # normalise ratings to [0, 1] to match Sigmoid output
    matrix = matrix / 10.0
    return matrix


def _bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalised Ranking loss.

    Maximises the probability that a positive item scores higher than a
    negative item for the same user.  Equivalent to minimising
    -log(sigmoid(pos - neg)).
    """
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()


def _sample_bpr_pairs(
    matrix_np: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple:
    """Sample (user_idx, pos_item_idx, neg_item_idx) triples for BPR training.

    A positive item is one the user has rated (matrix value > 0).
    A negative item is one the user has not rated (matrix value == 0).
    """
    n_users, n_items = matrix_np.shape

    user_indices = []
    pos_indices  = []
    neg_indices  = []

    per_user = max(1, n_samples // max(n_users, 1))

    for u in range(n_users):
        row      = matrix_np[u]
        pos_cols = np.where(row > 0)[0]
        neg_cols = np.where(row == 0)[0]
        if len(pos_cols) == 0 or len(neg_cols) == 0:
            continue
        k = min(per_user, len(pos_cols))
        chosen_pos = rng.choice(pos_cols, size=k, replace=False)
        chosen_neg = rng.choice(neg_cols,  size=k, replace=True)
        user_indices.extend([u] * k)
        pos_indices.extend(chosen_pos.tolist())
        neg_indices.extend(chosen_neg.tolist())

    return (
        np.array(user_indices, dtype=np.int64),
        np.array(pos_indices,  dtype=np.int64),
        np.array(neg_indices,  dtype=np.int64),
    )


def train_autoencoder(
    matrix: pd.DataFrame,
    epochs: int = 20,
    batch_size: int = 128,
    val_split: float = 0.2,
    patience: int = 3,
    pairs_per_user: int = 20,
) -> "AnimeAutoencoder":
    """Train the autoencoder with BPR (ranking) loss.

    Each mini-batch consists of BPR (user, pos_item, neg_item) triples drawn
    from the user-item matrix.  The AE encodes the full user row and the loss
    trains the decoder outputs so that the score for the positive item is
    higher than for the negative item — directly optimising ranking.

    Args:
        pairs_per_user: how many (pos, neg) pairs to sample per user per epoch.
                        Higher → more training signal per epoch, slower per epoch.
    """
    rng         = np.random.default_rng(SEED)
    matrix_np   = matrix.values.astype(np.float32)   # (n_users, n_items)
    n_users     = matrix_np.shape[0]
    X_tensor    = torch.tensor(matrix_np)

    model     = AnimeAutoencoder(matrix_np.shape[1])
    optimizer = torch.optim.Adam(model.parameters())

    n_pairs_total = n_users * pairs_per_user

    def _epoch_loss(idx_subset: np.ndarray, training: bool) -> float:
        """Run one pass over the given pair indices."""
        total_loss = 0.0
        count      = 0
        perm_sub   = rng.permutation(len(idx_subset)) if training else np.arange(len(idx_subset))
        for start in range(0, len(idx_subset), batch_size):
            batch = idx_subset[perm_sub[start: start + batch_size]]
            u_b   = torch.tensor(u_arr[batch],  dtype=torch.long)
            p_b   = torch.tensor(p_arr[batch],  dtype=torch.long)
            n_b   = torch.tensor(n_arr[batch],  dtype=torch.long)

            user_rows = X_tensor[u_b]          # (B, n_items)
            if training:
                model.train()
                optimizer.zero_grad()
                recon = model(user_rows)       # (B, n_items)
            else:
                model.eval()
                with torch.no_grad():
                    recon = model(user_rows)

            # gather scores for the specific pos/neg items
            pos_scores = recon[torch.arange(len(u_b)), p_b]   # (B,)
            neg_scores = recon[torch.arange(len(u_b)), n_b]   # (B,)
            loss = _bpr_loss(pos_scores, neg_scores)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(u_b)
            count      += len(u_b)

        return total_loss / max(count, 1)

    best_val_loss    = float("inf")
    patience_counter = 0
    best_state       = None

    for epoch in range(epochs):
        # re-sample pairs each epoch so the model sees fresh negatives
        u_arr, p_arr, n_arr = _sample_bpr_pairs(matrix_np, n_pairs_total, rng)
        perm    = rng.permutation(len(u_arr))
        tr_idx  = perm[:int(len(u_arr) * (1 - val_split))]
        vl_idx  = perm[int(len(u_arr) * (1 - val_split)):]

        train_loss = _epoch_loss(tr_idx, training=True)
        val_loss   = _epoch_loss(vl_idx, training=False)

        print(f"  Epoch {epoch+1:2d}/{epochs} — train_bpr: {train_loss:.6f}  val_bpr: {val_loss:.6f}")

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
