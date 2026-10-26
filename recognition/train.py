"""Training workflow for Facebook page classification.

This partial implementation provides model construction and a single training
step. Validation, experiment control, and checkpoint saving will be added in
later development steps.
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

    if model_name == "mlp":
        model = MLPBaseline(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout_p=dropout_p,
        )
    elif model_name == "gcn":
        model = TwoLayerGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout_p=dropout_p,
        )
    else:
        raise ValueError(
            f"Unknown model name: {model_name!r}"
        )

    return model


def forward_model(
    model: nn.Module,
    data: Data,
    model_name: str,
) -> torch.Tensor:
    """Run the correct forward pass for the selected model."""

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

    return outputs


def train_one_epoch(
    model: nn.Module,
    data: Data,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_name: str,
) -> float:
    """Train the selected model for one epoch."""

    model.train()

    outputs = forward_model(
        model,
        data,
        model_name,
    )

    loss = criterion(
        outputs[data.train_mask],
        data.y[data.train_mask],
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


def evaluate(
    model: nn.Module,
    data: Data,
    mask: torch.Tensor,
    model_name: str,
) -> dict[str, float]:
    """Evaluate the selected model on one dataset split."""

    raise NotImplementedError(
        "Model evaluation will be implemented next"
    )


def run_training(args: argparse.Namespace) -> None:
    """Run the complete training and validation experiment."""

    raise NotImplementedError(
        "The complete training experiment will be implemented later"
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
