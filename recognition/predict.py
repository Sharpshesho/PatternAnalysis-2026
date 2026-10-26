"""Prediction and evaluation for Facebook page classification.

This skeleton defines the planned steps for loading a trained model,
predicting page categories, calculating metrics, and exporting results.
The prediction and evaluation functions will be implemented later.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import Data


def load_trained_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Restore a model and its metadata from a training checkpoint."""

    raise NotImplementedError(
        "Checkpoint loading will be implemented next"
    )


def predict_scores(
    model: nn.Module,
    data: Data,
    model_name: str,
) -> torch.Tensor:
    """Return class scores using model.eval() and torch.no_grad()."""

    raise NotImplementedError(
        "The prediction forward pass will be implemented later"
    )


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
