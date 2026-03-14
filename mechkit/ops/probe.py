"""probe — train a linear probe on activations to test linear separability."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console

if TYPE_CHECKING:
    from mechkit.core.model import Model

console = Console()


def run_probe(
    model: "Model",
    texts: list[str],
    labels: list[int],
    *,
    at: str,
) -> dict[str, Any]:
    """Train a linear probe on activations at module *at*.

    Uses LogisticRegression from scikit-learn. Falls back to a simple
    torch-based probe if sklearn is not installed.
    """
    from mechkit.core.render import render_probe
    from mechkit.ops.activations import run_activations

    if len(texts) != len(labels):
        raise ValueError(f"texts ({len(texts)}) and labels ({len(labels)}) must have the same length.")

    # Extract activations for all texts
    features = []
    for text in texts:
        act = run_activations(model, text, at=at, print_stats=False)
        # Take last-token hidden state for sequence models
        if act.dim() == 3:
            vec = act[0, -1, :]  # (hidden,)
        elif act.dim() == 2:
            vec = act[-1, :]
        else:
            vec = act.view(-1)
        features.append(vec.cpu().float().numpy())

    import numpy as np

    X = np.stack(features)
    y = np.array(labels)

    try:
        result = _probe_sklearn(X, y)
    except ImportError:
        result = _probe_torch(X, y)

    result["module"] = at
    render_probe(result)
    return result


def _probe_sklearn(X: Any, y: Any) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    n_samples = len(y)

    if n_samples >= 10:
        cv_folds = min(5, n_samples)
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        scores = cross_val_score(clf, X, y, cv=cv_folds, scoring="accuracy")
        accuracy = float(scores.mean())
    else:
        accuracy = None

    # Train on full data for feature analysis
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X, y)
    train_accuracy = float(clf.score(X, y))

    # Top features by weight magnitude
    weights = clf.coef_[0] if clf.coef_.ndim == 2 else clf.coef_
    top_indices = list(reversed(sorted(range(len(weights)), key=lambda i: abs(weights[i]))))[:20]
    top_features = [(int(i), float(weights[i])) for i in top_indices]

    return {
        "accuracy": accuracy if accuracy is not None else train_accuracy,
        "train_accuracy": train_accuracy,
        "top_features": top_features,
    }


def _probe_torch(X: Any, y: Any) -> dict[str, Any]:
    """Fallback probe using pure PyTorch when sklearn is not available."""
    import numpy as np

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    n_features = X_t.shape[1]
    n_classes = len(set(y))

    linear = torch.nn.Linear(n_features, n_classes)
    optimizer = torch.optim.Adam(linear.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    linear.train()
    for _ in range(500):
        optimizer.zero_grad()
        logits = linear(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

    linear.eval()
    with torch.no_grad():
        preds = linear(X_t).argmax(dim=-1)
        train_accuracy = float((preds == y_t).float().mean().item())

    weights = linear.weight.detach()[0].numpy() if n_classes == 2 else linear.weight.detach().mean(dim=0).numpy()
    top_indices = list(reversed(sorted(range(len(weights)), key=lambda i: abs(weights[i]))))[:20]
    top_features = [(int(i), float(weights[i])) for i in top_indices]

    return {
        "accuracy": train_accuracy,
        "train_accuracy": train_accuracy,
        "top_features": top_features,
    }
