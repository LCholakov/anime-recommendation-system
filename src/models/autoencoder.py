import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

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


def _masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE only on non-zero entries (observed ratings)."""
    mask = (target > 0).float()
    diff = (pred - target) ** 2
    loss = (diff * mask).sum() / mask.sum().clamp(min=1)
    return loss


def build_user_item_matrix(train_df: pd.DataFrame) -> pd.DataFrame:
    matrix = train_df.pivot_table(
        index="user_id", columns="anime_id", values="rating", aggfunc="mean"
    ).fillna(0)
    # normalise ratings to [0, 1] to match Sigmoid output
    matrix = matrix / 10.0
    return matrix


def train_autoencoder(
    matrix: pd.DataFrame,
    epochs: int = 20,
    batch_size: int = 128,
    val_split: float = 0.2,
    patience: int = 3,
) -> AnimeAutoencoder:
    X = torch.tensor(matrix.values, dtype=torch.float32)
    dataset = TensorDataset(X)

    val_size   = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model     = AnimeAutoencoder(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters())

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for (batch,) in train_loader:
            optimizer.zero_grad()
            loss = _masked_mse(model(batch), batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_losses = [_masked_mse(model(b), b).item() for (b,) in val_loader]
        val_loss = sum(val_losses) / len(val_losses)

        print(f"  Epoch {epoch+1:2d}/{epochs} — val_loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
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
    model: AnimeAutoencoder,
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
