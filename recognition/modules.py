"""Neural-network models for Facebook page classification.

This initial skeleton defines the interfaces for a feature-only MLP baseline
and a two-layer Graph Convolutional Network. The layers and forward passes
will be implemented in later development steps.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """Classify each Facebook page using only its node features."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one class-score vector for every page."""

        raise NotImplementedError(
            "The MLP forward pass will be implemented next"
        )


class TwoLayerGCN(nn.Module):
    """Classify pages using node features and graph connections."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout_p = dropout_p

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return one class-score vector for every graph node."""

        raise NotImplementedError(
            "The GCN forward pass will be implemented later"
        )
