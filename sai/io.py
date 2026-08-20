"""Image loading utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as an HxWxC uint8 RGB array."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)
