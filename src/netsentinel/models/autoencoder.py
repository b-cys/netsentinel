"""Autoencoder anomaly detector — trained on BENIGN traffic only.

Because it never sees attacks during training, it can flag attack types
that did not exist in the training data (zero-day style detection).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from .ffnn import _device


class AE(nn.Module):
    def __init__(self, in_dim: int, hidden=(64, 32), latent: int = 8):
        super().__init__()
        enc, d = [], in_dim
        for h in hidden:
            enc += [nn.Linear(d, h), nn.ReLU()]
            d = h
        enc.append(nn.Linear(d, latent))
        dec, d = [], latent
        for h in reversed(hidden):
            dec += [nn.Linear(d, h), nn.ReLU()]
            d = h
        dec.append(nn.Linear(d, in_dim))
        self.encoder, self.decoder = nn.Sequential(*enc), nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderDetector:
    def __init__(self, hidden=(64, 32), latent=8, lr=1e-3, batch_size=1024, epochs=30,
                 patience=5, threshold_percentile=99.0, seed=42, verbose=True):
        self.hidden, self.latent, self.lr = hidden, latent, lr
        self.batch_size, self.epochs, self.patience = batch_size, epochs, patience
        self.threshold_percentile, self.seed, self.verbose = threshold_percentile, seed, verbose

    def fit(self, X_benign, X_val_benign):
        """Fit on benign only; threshold = percentile of benign validation error."""
        torch.manual_seed(self.seed)
        dev = _device()
        X = torch.as_tensor(np.asarray(X_benign), dtype=torch.float32)
        Xv = torch.as_tensor(np.asarray(X_val_benign), dtype=torch.float32).to(dev)
        self.model_ = AE(X.shape[1], self.hidden, self.latent).to(dev)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        mse = nn.MSELoss()

        best, best_state, bad = float("inf"), None, 0
        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            perm = torch.randperm(len(X))
            for i in range(0, len(X), self.batch_size):
                xb = X[perm[i:i + self.batch_size]].to(dev)
                opt.zero_grad()
                loss = mse(self.model_(xb), xb)
                loss.backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                val = mse(self.model_(Xv), Xv).item()
            if self.verbose:
                print(f"  epoch {epoch:02d}  val_recon={val:.5f}")
            if val < best - 1e-6:
                best, best_state, bad = val, copy.deepcopy(self.model_.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.model_.load_state_dict(best_state)
        self.threshold_ = float(np.percentile(self.score(X_val_benign),
                                              self.threshold_percentile))
        return self

    def score(self, X) -> np.ndarray:
        """Per-sample reconstruction error (higher = more anomalous)."""
        self.model_.eval()
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32).to(_device())
        with torch.no_grad():
            return ((self.model_(X) - X) ** 2).mean(dim=1).cpu().numpy()

    def predict(self, X) -> np.ndarray:
        return (self.score(X) > self.threshold_).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        """Pseudo-probability for ROC-AUC (monotonic in reconstruction error)."""
        s = self.score(X) / (self.threshold_ + 1e-12)
        p = s / (1 + s)
        return np.column_stack([1 - p, p])
