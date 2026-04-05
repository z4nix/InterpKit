"""probe — train a linear probe on activations to test linear separability."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console
from rich.progress import Progress

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()


def run_probe(
    model: Model,
    texts: list[str],
    labels: list[int],
    *,
    at: str,
) -> dict[str, Any]:
    """Train a linear probe on activations at module *at*.

    Uses LogisticRegression from scikit-learn. Falls back to a simple
    torch-based probe if sklearn is not installed.
    """
    from interpkit.core.render import render_probe
    from interpkit.ops.activations import run_activations

    if len(texts) != len(labels):
        raise ValueError(f"texts ({len(texts)}) and labels ({len(labels)}) must have the same length.")

    # Extract activations for all texts
    features = []
    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Extracting activations", total=len(texts))
        for text in texts:
            act = run_activations(model, text, at=at, print_stats=False)
            if act.dim() == 3:
                vec = act[0, -1, :]  # (hidden,)
            elif act.dim() == 2:
                vec = act[-1, :]
            else:
                vec = act.view(-1)
            features.append(vec.cpu().float().numpy())
            progress.advance(task)

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

    n_samples = len(y)

    if n_samples >= 20:
        from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        cv_folds = min(5, len(y_train))
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv_folds, scoring="accuracy")

        clf.fit(X_train, y_train)
        test_accuracy = float(clf.score(X_test, y_test))
        train_accuracy = float(clf.score(X_train, y_train))

        weights = clf.coef_[0] if clf.coef_.ndim == 2 else clf.coef_
        top_indices = list(reversed(sorted(range(len(weights)), key=lambda i: abs(weights[i]))))[:20]
        top_features = [(int(i), float(weights[i])) for i in top_indices]

        return {
            "accuracy": test_accuracy,
            "cv_accuracy": float(cv_scores.mean()),
            "train_accuracy": train_accuracy,
            "eval_method": "holdout",
            "top_features": top_features,
        }

    if n_samples >= 10:
        from sklearn.model_selection import cross_val_score

        cv_folds = min(5, n_samples)
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        scores = cross_val_score(clf, X, y, cv=cv_folds, scoring="accuracy")

        clf.fit(X, y)
        weights = clf.coef_[0] if clf.coef_.ndim == 2 else clf.coef_
        top_indices = list(reversed(sorted(range(len(weights)), key=lambda i: abs(weights[i]))))[:20]
        top_features = [(int(i), float(weights[i])) for i in top_indices]

        return {
            "accuracy": float(scores.mean()),
            "eval_method": "cv_only",
            "top_features": top_features,
        }

    # < 10 samples: train-only, no reliable evaluation possible
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X, y)
    train_accuracy = float(clf.score(X, y))

    weights = clf.coef_[0] if clf.coef_.ndim == 2 else clf.coef_
    top_indices = list(reversed(sorted(range(len(weights)), key=lambda i: abs(weights[i]))))[:20]
    top_features = [(int(i), float(weights[i])) for i in top_indices]

    return {
        "accuracy": train_accuracy,
        "eval_method": "train_only",
        "top_features": top_features,
    }


def _probe_torch(X: Any, y: Any) -> dict[str, Any]:
    """Fallback probe using pure PyTorch when sklearn is not available."""

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    n_features = X_t.shape[1]
    n_classes = len(set(y))

    linear = torch.nn.Linear(n_features, n_classes)
    optimizer = torch.optim.Adam(linear.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    n_epochs = 500
    linear.train()
    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Training probe", total=n_epochs)
        for _ in range(n_epochs):
            optimizer.zero_grad()
            logits = linear(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()
            progress.advance(task)

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
