"""Cross-image consistency signal: batch-level variance detection.

AI-generated image series (Instagram carousels, batch generations from the
same prompt) have extremely uniform color and texture statistics across
images. Real photo sets have high variance due to different lighting,
angles, subjects, and camera conditions.

This signal operates on a BATCH of images, not a single image. It measures:

1. **Color statistics variance**: Mean RGB across the batch. AI batches
   have stddev < 5 across images; real photo sets have stddev > 15.

2. **Edge density variance**: Edge density should vary across real photos.
   AI-generated series have near-constant edge density.

3. **Histogram correlation**: Pairwise histogram correlation across images.
   AI batches have high correlation (>0.8); real photos have low (<0.5).

This signal is only available when detecting multiple images together.
For single-image detection, it returns a neutral score with low weight.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import sobel

from sai.signals.base import Signal, SignalResult


class CrossImageConsistencySignal(Signal):
    name = "cross_image"

    def __init__(
        self,
        color_std_threshold: float = 8.0,
        edge_std_threshold: float = 0.03,
        hist_corr_threshold: float = 0.75,
    ) -> None:
        self.color_std_threshold = color_std_threshold
        self.edge_std_threshold = edge_std_threshold
        self.hist_corr_threshold = hist_corr_threshold

    def _mean_rgb(self, image: np.ndarray) -> np.ndarray:
        return np.array(image.mean(axis=(0, 1)))

    def _edge_density(self, image: np.ndarray) -> float:
        gray = np.mean(image, axis=2).astype(np.float32)
        edges = sobel(gray)
        return float(np.mean(np.abs(edges) > 30))

    def _color_hist(self, image: np.ndarray, bins: int = 32) -> np.ndarray:
        """Normalized color histogram for correlation comparison."""
        hists = []
        for c in range(3):
            hist, _ = np.histogram(image[:, :, c], bins=bins, range=(0, 256))
            hist = hist / (hist.sum() + 1e-9)
            hists.append(hist)
        return np.concatenate(hists)

    def analyze(self, image: np.ndarray) -> SignalResult:
        """Single-image analysis: return neutral with low weight."""
        return SignalResult(
            score=0.5,
            weight=0.05,
            features={"note": "single image - cross-image analysis requires batch"},
        )

    def analyze_batch(self, images: list[np.ndarray]) -> list[SignalResult]:
        """Analyze a batch of images. Returns one SignalResult per image.

        All images get the same score (the batch-level AI probability) but
        individual weights may vary if an image is an outlier.
        """
        if len(images) < 2:
            return [self.analyze(img) for img in images]

        # Compute per-image features
        mean_rgbs = [self._mean_rgb(img) for img in images]
        edge_densities = [self._edge_density(img) for img in images]
        hists = [self._color_hist(img) for img in images]

        # 1. Color statistics variance across batch
        color_std = float(np.std(mean_rgbs, axis=0).mean())
        # Low color_std -> uniform -> AI. High -> varied -> real.
        color_score = 1.0 / (1.0 + np.exp(-0.5 * (self.color_std_threshold - color_std)))

        # 2. Edge density variance across batch
        edge_std = float(np.std(edge_densities))
        edge_score = 1.0 / (1.0 + np.exp(-50.0 * (self.edge_std_threshold - edge_std)))

        # 3. Pairwise histogram correlation
        n = len(hists)
        corrs = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = hists[i], hists[j]
                a_c = a - a.mean()
                b_c = b - b.mean()
                denom = (np.linalg.norm(a_c) * np.linalg.norm(b_c)) + 1e-9
                corrs.append(float(np.dot(a_c, b_c) / denom))
        mean_corr = float(np.mean(corrs)) if corrs else 0.5
        # High correlation -> AI. Low -> real.
        corr_score = 1.0 / (1.0 + np.exp(-15.0 * (mean_corr - self.hist_corr_threshold)))

        # Combined batch score
        batch_score = float(np.clip(
            0.4 * color_score + 0.3 * edge_score + 0.3 * corr_score,
            0.0, 1.0
        ))

        # Weight: higher when batch is larger and more uniform
        # If batch is clearly uniform (high batch_score), weight is high
        # If batch is clearly varied (low batch_score), weight is also high (confident real)
        # If batch is ambiguous, weight is lower
        confidence = abs(batch_score - 0.5) * 2.0  # 0 at 0.5, 1 at 0 or 1
        weight = float(np.clip(0.3 + 0.5 * confidence, 0.1, 0.8))

        results = []
        for i, img in enumerate(images):
            # Check if this image is an outlier in the batch
            color_dist = float(np.linalg.norm(mean_rgbs[i] - np.mean(mean_rgbs, axis=0)))
            is_outlier = color_dist > 2 * np.std([np.linalg.norm(m - np.mean(mean_rgbs, axis=0)) for m in mean_rgbs])

            # Outliers get lower weight (they don't represent the batch)
            img_weight = weight * 0.5 if is_outlier else weight

            results.append(SignalResult(
                score=batch_score,
                weight=img_weight,
                features={
                    "batch_size": len(images),
                    "batch_color_std": color_std,
                    "batch_edge_std": edge_std,
                    "batch_hist_corr": mean_corr,
                    "batch_color_score": float(color_score),
                    "batch_edge_score": float(edge_score),
                    "batch_corr_score": float(corr_score),
                    "is_outlier": bool(is_outlier),
                    "image_index": i,
                },
            ))

        return results
