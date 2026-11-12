"""Feed-forward neural network (PyTorch) with an sklearn-like interface."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(256, 128, 64), dropout: float = 0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FFNNClassifier:
    """Binary classifier: class-weighted BCE + early stopping on validation loss."""

    def __init__(self, hidden=(256, 128, 64), dropout=0.2, lr=1e-3, batch_size=1024,
                 epochs=30, patience=5, seed=42, verbose=True, adv=None):
        # adv = {"mask": bool array, "eps": float, "steps": int} turns on adversarial
        # training: half of the attack flows in every batch are replaced by PGD
        # evasion attempts against the current model.
        self.adv = adv
        self.hidden, self.dropout, self.lr = hidden, dropout, lr
        self.batch_size, self.epochs, self.patience = batch_size, epochs, patience
        self.seed, self.verbose = seed, verbose
        self.history: list[dict] = []

    def fit(self, X, y, X_val, y_val):
        torch.manual_seed(self.seed)
        dev = _device()
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        Xv = torch.as_tensor(np.asarray(X_val), dtype=torch.float32).to(dev)
        yv = torch.as_tensor(np.asarray(y_val), dtype=torch.float32).to(dev)

        self.model_ = MLP(X.shape[1], self.hidden, self.dropout).to(dev)
        n_pos = float(y.sum())
        pos_weight = torch.tensor((len(y) - n_pos) / max(n_pos, 1.0), device=dev)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)

        best, best_state, bad = float("inf"), None, 0
        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            perm = torch.randperm(len(X))
            total = 0.0
            for i in range(0, len(X), self.batch_size):
                b = perm[i:i + self.batch_size]
                xb, yb = X[b].to(dev), y[b].to(dev)
                if self.adv is not None:
                    from ..adversarial import pgd
                    pick = (yb == 1) & (torch.rand(len(yb), device=dev) < 0.5)
                    if pick.any():
                        mask = torch.as_tensor(self.adv["mask"], dtype=torch.float32, device=dev)
                        xb = xb.clone()
                        xb[pick] = pgd(self.model_, xb[pick], mask, self.adv.get("eps", 0.5),
                                       self.adv.get("steps", 5))
                opt.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(Xv), yv).item()
            self.history.append({"epoch": epoch, "train_loss": total / len(X),
                                 "val_loss": val_loss})
            if self.verbose:
                print(f"  epoch {epoch:02d}  train={total / len(X):.4f}  val={val_loss:.4f}")
            if val_loss < best - 1e-5:
                best, best_state, bad = val_loss, copy.deepcopy(self.model_.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X) -> np.ndarray:
        self.model_.eval()
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32).to(_device())
        with torch.no_grad():
            p = torch.sigmoid(self.model_(X)).cpu().numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
