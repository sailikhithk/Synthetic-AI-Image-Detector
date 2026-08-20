"""Noise residual signal (PRNU-style).

Real cameras leave a Photo Response Non-Uniformity (PRNU) noise fingerprint
from sensor demosaicing and lens distortion. Synthetic images lack this
fingerprint: their "noise" is generator-internal and lacks the spatial
structure of a real sensor pattern.

We extract a high-pass residual via a Laplacian-of-Gaussian, then measure:
1. Spatial consistency: real PRNU has stable spatial statistics; AI noise
   is more uniform/random.
2. Noise variance per channel: real cameras have channel-dependent noise
   (Bayer pattern), AI generators typically do not.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from sai.signals.base import Signal, SignalResult


class NoiseResidualSignal(Signal):
    name = "noise_residual"

    def __init__(self, sigma: float = 1.0) -> None:
        self.sigma = sigma

    def _residual(self, channel: np.ndarray) -> np.ndarray:
        f = channel.astype(np.float32) / 255.0
        blurred = gaussian_filter(f, sigma=self.sigma)
        return f - blurred

    def analyze(self, image: np.ndarray) -> SignalResult:
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        residuals = [self._residual(image[:, :, c]) for c in range(image.shape[2])]

        # Per-channel variance: real cameras have channel-dependent noise
        variances = [float(np.var(r)) for r in residuals]
        var_spread = float(np.std(variances))
        # Real images: var_spread > 0 (Bayer asymmetry). AI: closer to 0.
        # Low var_spread -> high score (AI-like). Thresholds for [0,1] residuals.
        asym_logit = float(np.clip(5000.0 * (1e-4 - var_spread), -50.0, 50.0))
        channel_asym_score = 1.0 / (1.0 + np.exp(-asym_logit))

        # Spatial consistency: std of local std across the residual.
        # Real PRNU has spatial structure (more variation in local std).
        # AI noise is more spatially uniform (less variation in local std).
        local_stds = []
        for r in residuals:
            h, w = r.shape
            # 16x16 block local std
            bh, bw = max(1, h // 16), max(1, w // 16)
            for i in range(0, h - bh + 1, bh):
                for j in range(0, w - bw + 1, bw):
                    local_stds.append(float(np.std(r[i : i + bh, j : j + bw])))
        spatial_var = float(np.std(local_stds)) if local_stds else 0.0
        # Low spatial_var -> uniform noise -> AI
        spatial_logit = float(np.clip(500.0 * (2e-3 - spatial_var), -50.0, 50.0))
        spatial_score = 1.0 / (1.0 + np.exp(-spatial_logit))

        combined = float(np.clip(0.5 * channel_asym_score + 0.5 * spatial_score, 0.0, 1.0))

        return SignalResult(
            score=combined,
            weight=0.6,
            features={
                "var_per_channel": variances,
                "var_spread": var_spread,
                "spatial_var": spatial_var,
                "channel_asym_score": float(channel_asym_score),
                "spatial_score": float(spatial_score),
            },
        )
