"""Training workflow for Facebook page classification.

This initial skeleton defines the training interface and experiment settings.
The training loop, validation procedure, checkpoint saving, and final
evaluation will be implemented in later steps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data

from dataset import load_facebook_page_page
from modules import MLPBaseline, TwoLayerGCN


def set_seed(seed: int) -> None:
    """Set the random seed so experiments can be reproduced."""

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(
    model_name: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    dropout_p: float,
) -> nn.Module:
    """Create either the MLP baseline or the two-layer GCN."""

    raise NotImplementedError(
        "Model creation will be implemented next"
    )


def train_one_epoch(
    model: nn.Module,
    data: Data,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_name: str,
) -> float:
    """Train the selected model for one epoch."""

    raise NotImplementedError(
        "The training step will be implemented later"
    )


def evaluate(
    model: nn.Module,
    data: Data,
    mask: torch.Tensor,
    model_name: str,
) -> dict[str, float]:
    """Evaluate the selected model on one dataset split."""

    raise NotImplementedError(
        "Model evaluation will be implemented later"
    )


def run_training(args: argparse.Namespace) -> None:
    """Run the complete training and validation experiment."""

    raise NotImplementedError(
        "The complete experiment will be implemented later"
    )


def parse_args() -> argparse.Namespace:
    """Read experiment settings from the command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Train an MLP or GCN on the Facebook Page-Page dataset"
        )
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to facebook.npz",
    )
    parser.add_argument(
        "--model",
        choices=("mlp", "gcn"),
        required=True,
        help="Model architecture to train",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
    )

    return parser.parse_args()


def main() -> None:
    """Run the training program."""

    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
