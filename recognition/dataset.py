"""Dataset loading and split construction for Facebook page classification.

This initial implementation validates the dataset archive and split settings.
Graph conversion and leakage-free split creation will be implemented next.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SPLIT_RATIOS = (0.60, 0.20, 0.20)
REQUIRED_ARRAYS = ("X", "edges_x", "edges_y", "y", "page_name")


def _validate_ratios(
    ratios: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Check that three positive split ratios add up to one."""

    if len(ratios) != 3:
        raise ValueError(
            "ratios must contain train, validation, and test values"
        )

    values = tuple(float(value) for value in ratios)

    if any(value <= 0.0 for value in values):
        raise ValueError("all split ratios must be positive")

    if not np.isclose(sum(values), 1.0):
        raise ValueError("split ratios must add up to 1.0")

    return values


def _load_archive(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Load the required arrays from the Facebook Page-Page archive."""

    path = Path(npz_path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"Dataset archive was not found: {path}")

    if path.suffix.lower() != ".npz":
        raise ValueError("Dataset archive must be an .npz file")

    try:
        with np.load(path, allow_pickle=True) as archive:
            missing = [
                name
                for name in REQUIRED_ARRAYS
                if name not in archive.files
            ]

            if missing:
                raise ValueError(
                    f"Dataset archive is missing required arrays: {missing}"
                )

            arrays = {
                name: archive[name]
                for name in REQUIRED_ARRAYS
            }

    except OSError as exc:
        raise ValueError(
            f"Could not read dataset archive: {path}"
        ) from exc

    return arrays


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

    _validate_ratios(ratios)
    _load_archive(npz_path)

    raise NotImplementedError(
        "Graph validation, tensor conversion, and split masks come next"
    )
