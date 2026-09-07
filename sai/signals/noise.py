"""Noise residual signal (PRNU-style) with JPEG compression awareness.

Real cameras leave a Photo Response Non-Uniformity (PRNU) noise fingerprint
from sensor demosaicing and lens distortion. Synthetic images lack this
fingerprint: their "noise" is generator-internal and lacks the spatial
structure of a real sensor pattern.

We extract a high-pass residual via a Laplacian-of-Gaussian, then measure:
1. Spatial consistency: real PRNU has stable spatial statistics; AI noise
   is more uniform/random.
2. Noise variance per channel: real cameras have channel-dependent noise
   (Bayer pattern), AI generators typically do not.
3. JPEG blockiness: platform-re-encoded (Instagram, Twitter) images have
   8x8 DCT block artifacts that mimic "real" noise structure. We detect
   and compensate for this to avoid false "real" verdicts on AI images
   that were uploaded to social media.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from sai.signals.base import Signal, SignalResult


class NoiseResidualSignal(Signal):
    name = "noise_residual"

    def __init__(self, sigma: float = 1.0, jpeg_block_threshold: float = 0.3) -> None:
        self.sigma = sigma
        self.jpeg_block_threshold = jpeg_block_threshold

    def _residual(self, channel: np.ndarray) -> np.ndarray:
        f = channel.astype(np.float32) / 255.0
        blurred = gaussian_filter(f, sigma=self.sigma)
        return f - blurred

    def _detect_jpeg_blockiness(self, image: np.ndarray) -> float:
        """Detect 8x8 JPEG block artifacts. Returns 0..1 (1 = strong JPEG)."""
        gray = np.mean(image, axis=2).astype(np.float32) / 255.0
        h, w = gray.shape
        # Measure discontinuities at 8-pixel boundaries
        block_boundary_diffs = []
        for i in range(8, h - 8, 8):
            row_diff = np.mean(np.abs(gray[i, :] - gray[i - 1, :]))
            interior_diff = np.mean(np.abs(gray[i + 1, :] - gray[i, :]))
            block_boundary_diffs.append(row_diff / (interior_diff + 1e-9))
        for j in range(8, w - 8, 8):
            col_diff = np.mean(np.abs(gray[:, j] - gray[:, j - 1]))
            interior_diff = np.mean(np.abs(gray[:, j + 1] - gray[:, j]))
            block_boundary_diffs.append(col_diff / (interior_diff + 1e-9))
        if not block_boundary_diffs:
            return 0.0
        # Ratio > 1.0 means block boundaries have more discontinuity than interior
        # Higher ratio = stronger JPEG compression
        mean_ratio = float(np.mean(block_boundary_diffs))
        # Map to [0, 1]: ratio of 1.0 = no JPEG, ratio of 1.5+ = strong JPEG
        return float(np.clip((mean_ratio - 1.0) / 0.5, 0.0, 1.0))

    def analyze(self, image: np.ndarray) -> SignalResult:
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # Detect JPEG blockiness - if strong, reduce weight of this signal
        # because JPEG compression artifacts mimic PRNU spatial structure
        jpeg_blockiness = self._detect_jpeg_blockiness(image)
        is_jpeg_compressed = jpeg_blockiness > self.jpeg_block_threshold

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

        # JPEG compensation: if image is JPEG-compressed, the noise signal
        # is unreliable because JPEG block artifacts create false spatial
        # structure. Reduce weight and pull score toward neutral (0.5).
        if is_jpeg_compressed:
            # Blend toward neutral based on JPEG strength
            jpeg_factor = float(np.clip(jpeg_blockiness, 0.0, 1.0))
            combined = float(0.5 * jpeg_factor + combined * (1.0 - jpeg_factor))
            weight = 0.6 * (1.0 - jpeg_factor * 0.7)  # reduce weight up to 70%
        else:
            weight = 0.6

        return SignalResult(
            score=combined,
            weight=weight,
            features={
                "var_per_channel": variances,
                "var_spread": var_spread,
                "spatial_var": spatial_var,
                "channel_asym_score": float(channel_asym_score),
                "spatial_score": float(spatial_score),
                "jpeg_blockiness": float(jpeg_blockiness),
                "is_jpeg_compressed": bool(is_jpeg_compressed),
                "weight_adjusted": bool(is_jpeg_compressed),
            },
        )
