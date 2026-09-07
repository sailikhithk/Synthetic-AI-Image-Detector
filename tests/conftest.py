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


def make_ai_batch(n: int = 5, size: int = 128, seed: int = 0) -> list:
    """Generate a batch of AI-like images with very uniform statistics.

    AI-generated series (carousels, batch gens) have near-identical color stats.
    """
    images = []
    for i in range(n):
        # Same base gradient, tiny perturbation -> uniform across batch
        rng = np.random.default_rng(seed + i)
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        base = np.outer(y, x)
        img = np.stack([base, base, base], axis=-1)
        # Very small noise so batch stats are nearly identical
        noise = rng.normal(0, 0.002, img.shape)
        img = img + noise
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        images.append(img)
    return images


def make_real_batch(n: int = 5, size: int = 128, seed: int = 0) -> list:
    """Generate a batch of real-like images with varied statistics.

    Real photo sets have high variance due to different lighting, angles, subjects.
    """
    images = []
    for i in range(n):
        rng = np.random.default_rng(seed + i * 100)
        # Different base brightness/contrast per image (varied lighting)
        brightness = 0.3 + 0.7 * rng.random()
        contrast = 0.5 + 1.5 * rng.random()
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        base = np.outer(y, x) * contrast + brightness
        # Different channel scaling per image (varied white balance)
        r_scale = 0.8 + 0.4 * rng.random()
        g_scale = 0.8 + 0.4 * rng.random()
        b_scale = 0.8 + 0.4 * rng.random()
        img = np.stack([base * r_scale, base * g_scale, base * b_scale], axis=-1)
        # Bayer-style noise with varied intensity
        noise = rng.normal(0, 0.02 + 0.03 * rng.random(), img.shape)
        noise[:, :, 1] *= 1.6
        noise[:, :, 2] *= 0.7
        img = img + noise
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        images.append(img)
    return images
