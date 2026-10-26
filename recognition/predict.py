"""Checkpoint loading and prediction for Facebook page classification.

This partial version restores a trained MLP or GCN and returns class scores.
Evaluation metrics, result exports, and the full workflow remain unfinished.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
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

    raise NotImplementedError(
        "Evaluation metrics will be implemented later"
    )


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
