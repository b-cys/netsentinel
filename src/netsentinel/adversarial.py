"""Evasion attacks against the detector, and adversarial training as a defence.

Threat model: the attacker already runs an attack and wants it classified as
benign. They can only change what they control on their own side of the
connection, and only in ways that keep the attack working:

* only *mutable* features move (forward packet sizes, forward timing, duration,
  forward packet counts: things padding, delays and dummy packets can change);
* those features can only go *up* (you can pad or slow down, not un-send bytes);
* the change is bounded by ``eps`` in the model's standardised input space.

This is a feature-space attack. It is a realistic lower bound on robustness,
not proof that a matching packet trace exists.
"""
from __future__ import annotations

import numpy as np
import torch

from .models.ffnn import _device


def mutable_mask(feature_names: list[str], keywords: list[str]) -> np.ndarray:
    """Boolean mask of features whose name contains one of ``keywords``."""
    low = [k.lower() for k in keywords]
    return np.array([any(k in f.lower() for k in low) for f in feature_names])


def pgd(net: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor, eps: float = 0.5,
        steps: int = 20, increase_only: bool = True) -> torch.Tensor:
    """Projected gradient descent that pushes attack flows towards "benign"."""
    was_training = net.training
    net.eval()
    lo = 0.0 if increase_only else -eps
    delta = torch.zeros_like(x)
    alpha = 2.5 * eps / steps
    for _ in range(steps):
        delta.requires_grad_(True)
        logit = net(x + delta * mask)
        grad, = torch.autograd.grad(logit.sum(), delta)
        delta = (delta.detach() - alpha * grad.sign()).clamp(lo, eps)
    net.train(was_training)
    return (x + delta * mask).detach()


def evade(model, X: np.ndarray, mask: np.ndarray, eps: float = 0.5, steps: int = 20,
          increase_only: bool = True, batch_size: int = 4096) -> np.ndarray:
    """Adversarial versions of the attack flows ``X`` against an FFNNClassifier."""
    dev = _device()
    m = torch.as_tensor(mask, dtype=torch.float32, device=dev)
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(np.asarray(X[i:i + batch_size]), dtype=torch.float32, device=dev)
        out.append(pgd(model.model_, xb, m, eps, steps, increase_only).cpu().numpy())
    return np.concatenate(out) if out else np.empty_like(X)
