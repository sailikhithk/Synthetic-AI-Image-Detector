"""Synthetic image fixtures for tests.

We construct deterministic 'real-camera-like' and 'AI-like' images so the
tests do not need a real dataset. Real images get a Bayer-style channel
asymmetry and high-frequency sensor noise; AI images get smooth
generator-style noise and uniform cross-channel spectra.
"""

from __future__ import annotations

import numpy as np


def make_real_like(size: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # smooth gradient base (low freq)
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    base = np.outer(y, x)
    img = np.stack([base, base * 0.9, base * 1.1], axis=-1)
    # Bayer-style channel-dependent noise
    noise = rng.normal(0, 0.03, img.shape)
    noise[:, :, 1] *= 1.6  # green channel stronger (Bayer G1G2)
    noise[:, :, 2] *= 0.7  # red weaker
    img = img + noise
    # add high-frequency sensor texture
    hf = rng.normal(0, 0.01, img.shape)
    img = img + hf
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def make_ai_like(size: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    base = np.outer(y, x)
    # perfectly correlated channels (synthesis signature)
    img = np.stack([base, base, base], axis=-1)
    # uniform, low-variance noise across channels
    noise = rng.normal(0, 0.005, img.shape)
    img = img + noise
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img
