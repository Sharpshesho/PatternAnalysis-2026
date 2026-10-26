"""Prediction and metric calculations for Facebook page classification.

This partial version restores a trained MLP or GCN, returns class scores,
and evaluates a selected split. Exports and the full workflow remain unfinished.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch_geometric.data import Data

from train import create_model, load_checkpoint


def load_trained_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Restore a model and its metadata from a training checkpoint."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint was not found: {checkpoint_path}"
        )

    checkpoint = load_checkpoint(checkpoint_path, device)

    required = {"model_config", "seed", "class_values"}
    missing = required.difference(checkpoint)

    if missing:
        raise ValueError(
            f"Checkpoint is missing metadata: {sorted(missing)}"
        )

    model = create_model(
        **checkpoint["model_config"]
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, checkpoint


def predict_scores(
    model: nn.Module,
    data: Data,
    model_name: str,
) -> torch.Tensor:
    """Return class scores using model.eval() and torch.no_grad()."""

    model.eval()

    with torch.no_grad():
        if model_name == "mlp":
            outputs = model(data.x)
        elif model_name == "gcn":
            outputs = model(
                data.x,
                data.edge_index,
            )
        else:
            raise ValueError(
                f"Unknown model name: {model_name!r}"
            )

    if not bool(torch.isfinite(outputs).all()):
        raise ValueError(
            "Model predictions contain NaN or infinite values"
        )

    return outputs


def calculate_metrics(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    """Calculate accuracy, macro-F1, AUROC, and a confusion matrix."""

    if outputs.ndim != 2 or labels.ndim != 1:
        raise ValueError(
            "Expected outputs=[nodes, classes] and labels=[nodes]"
        )

    if mask.dtype != torch.bool or mask.shape != labels.shape:
        raise ValueError(
            "The split mask must be Boolean with one value per node"
        )

    if outputs.shape[0] != labels.numel() or not bool(mask.any()):
        raise ValueError(
            "Predictions must match labels and the split cannot be empty"
        )

    selected_outputs = outputs[mask].detach().cpu()
    y_true = labels[mask].detach().cpu().numpy()
    probabilities = selected_outputs.softmax(dim=1).numpy()
    y_pred = probabilities.argmax(axis=1)
    class_ids = list(range(outputs.shape[1]))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_ids,
        zero_division=0,
    )

    # Evaluate each class against all remaining classes.
    per_class_auroc: list[float | None] = []

    for class_index in class_ids:
        binary_labels = (y_true == class_index).astype(np.int64)

        if np.unique(binary_labels).size < 2:
            per_class_auroc.append(None)
        else:
            class_auroc = roc_auc_score(
                binary_labels,
                probabilities[:, class_index],
            )
            per_class_auroc.append(float(class_auroc))

    # A complete macro AUROC needs positive and negative examples per class.
    macro_auroc = None

    if all(score is not None for score in per_class_auroc):
        macro_auroc = float(np.mean(per_class_auroc))

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=class_ids,
    )

    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )

    return {
        "count": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1.mean()),
        "macro_auroc_ovr": macro_auroc,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
        "per_class_auroc_ovr": per_class_auroc,
        "confusion_matrix": matrix.tolist(),
        "normalized_confusion_matrix": normalized_matrix.tolist(),
    }


def save_prediction_outputs(
    output_dir: Path,
    data: Data,
    outputs: torch.Tensor,
    mask: torch.Tensor,
    metrics: dict[str, Any],
) -> None:
    """Save metrics, predictions, failure examples, and confusion plots."""

    raise NotImplementedError(
        "Prediction exports will be implemented later"
    )


def run_prediction(args: argparse.Namespace) -> None:
    """Load the model, recreate its seeded split, and evaluate predictions."""

    raise NotImplementedError(
        "The complete prediction workflow will be implemented later"
    )


def parse_args() -> argparse.Namespace:
    """Read prediction settings from the command line."""

    parser = argparse.ArgumentParser(
        description="Evaluate a trained Facebook page classifier"
    )
    parser.add_argument(
        "--data-path",
        "--data",
        dest="data_path",
        type=Path,
        required=True,
        help="Path to facebook.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="test",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device selection: auto, cpu, cuda, or cuda:N",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/predictions"),
    )

    return parser.parse_args()


def main() -> None:
    """Run the prediction program."""

    args = parse_args()
    run_prediction(args)


if __name__ == "__main__":
    main()
