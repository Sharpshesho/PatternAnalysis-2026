"""Dataset loading and split construction for Facebook page classification.

This initial skeleton records the public interface only. Data validation,
preprocessing, and leakage-free split creation will be implemented after the
dataset audit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_SPLIT_RATIOS = (0.60, 0.20, 0.20)


def load_facebook_page_page(
    npz_path: str | Path,
    seed: int = 42,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> Any:
    """Load the graph and return features, edges, labels, and split masks.

    Args:
        npz_path: Location of the local Facebook Page-Page archive.
        seed: Random seed used to reproduce the data split.
        ratios: Train, validation, and test proportions.

    Returns:
        A graph data object containing node features, edges, labels, and three
        mutually exclusive Boolean masks.
    """

    raise NotImplementedError("Dataset loading will be implemented next")
