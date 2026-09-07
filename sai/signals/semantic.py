"""Semantic signal: color distribution and texture uniformity analysis.

Pixel-level statistical signals (frequency, noise, reconstruction) miss
semantic-level artifacts that are obvious to humans:

1. **Color distribution uniformity**: AI-generated image series (carousels,
   batch generations) have extremely uniform color statistics across images.
   Real photo sets have high variance due to different lighting, angles, and
   subjects. This signal measures the local color histogram concentration.

2. **Edge density uniformity**: AI-generated "handwritten" or "diagram" style
   images have very consistent edge density. Real photos vary widely.

3. **Color palette size**: AI images often have a limited, synthetic color
   palette (few unique colors relative to image size). Real photos have
   continuous color distributions from sensor noise and natural lighting.

4. **Spatial smoothness**: AI-generated images often have unnaturally smooth
   gradients in areas where real photos would have texture/noise. This measures
   the ratio of smooth regions to textured regions.

5. **Channel correlation**: AI generators synthesize RGB channels jointly,
   producing higher channel-to-channel correlation than real camera images
   (which have Bayer-pattern demosaicing differences).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import sobel

from sai.signals.base import Signal, SignalResult


class SemanticSignal(Signal):
    name = "semantic"

    def __init__(
        self,
        palette_ratio_threshold: float = 0.15,
        smooth_ratio_threshold: float = 0.45,
        channel_corr_threshold: float = 0.92,
        edge_density_range: tuple = (0.05, 0.30),
        dominant_color_threshold: float = 0.35,
    ) -> None:
        self.palette_ratio_threshold = palette_ratio_threshold
        self.smooth_ratio_threshold = smooth_ratio_threshold
        self.channel_corr_threshold = channel_corr_threshold
        self.edge_density_range = edge_density_range
        self.dominant_color_threshold = dominant_color_threshold

    def _color_palette_size(self, image: np.ndarray) -> float:
        """Ratio of unique colors to total pixels. Low = synthetic."""
        # Downsample for speed
        h, w = image.shape[:2]
        step_h = max(1, h // 256)
        step_w = max(1, w // 256)
        small = image[::step_h, ::step_w]
        total = small.shape[0] * small.shape[1]
        # Quantize to reduce color count (4 bits per channel)
        quantized = (small // 16) * 16
        unique = len(np.unique(quantized.reshape(-1, 3), axis=0))
        return unique / (total / 16)  # normalized: 1.0 = many colors, <0.15 = synthetic

    def _edge_density(self, image: np.ndarray) -> float:
        """Fraction of pixels that are edges. AI diagrams have uniform edge density."""
        gray = np.mean(image, axis=2).astype(np.float32)
        edges = sobel(gray)
        return float(np.mean(np.abs(edges) > 30))

    def _smooth_ratio(self, image: np.ndarray) -> float:
        """Fraction of 16x16 blocks that are smooth (low variance). AI images have more."""
        gray = np.mean(image, axis=2).astype(np.float32)
        h, w = gray.shape
        bh, bw = max(1, h // 16), max(1, w // 16)
        smooth_count = 0
        total_blocks = 0
        for i in range(0, h - bh + 1, bh):
            for j in range(0, w - bw + 1, bw):
                block = gray[i:i + bh, j:j + bw]
                if np.std(block) < 5.0:
                    smooth_count += 1
                total_blocks += 1
        return smooth_count / max(total_blocks, 1)

    def _channel_correlation(self, image: np.ndarray) -> float:
        """Mean pairwise correlation between RGB channels. High = synthetic."""
        r = image[:, :, 0].astype(np.float32).flatten()
        g = image[:, :, 1].astype(np.float32).flatten()
        b = image[:, :, 2].astype(np.float32).flatten()
        corrs = []
        for a, c in [(r, g), (r, b), (g, b)]:
            a_c = a - a.mean()
            c_c = c - c.mean()
            denom = (np.linalg.norm(a_c) * np.linalg.norm(c_c)) + 1e-9
            corrs.append(float(np.dot(a_c, c_c) / denom))
        return float(np.mean(corrs))

    def _gradient_smoothness(self, image: np.ndarray) -> float:
        """Measure gradient smoothness. AI images have unnaturally smooth gradients."""
        gray = np.mean(image, axis=2).astype(np.float32)
        # Compute gradient
        gy = np.diff(gray, axis=0)
        gx = np.diff(gray, axis=1)
        # Measure how many gradient values are near-zero (smooth regions)
        g_mag = np.sqrt(gy[:, :-1]**2 + gx[:-1, :]**2)
        near_zero = float(np.mean(g_mag < 1.0))
        return near_zero

    def _dominant_color_concentration(self, image: np.ndarray) -> float:
        """Fraction of pixels that match the most common quantized color.

        AI-generated images with solid/gradient backgrounds have a very high
        concentration of one dominant color (often >50% for dark-background
        study notes). Real photos have much more distributed color histograms.
        """
        # Quantize to 4 bits per channel (16 levels = 4096 possible colors)
        quantized = (image // 16) * 16
        # Find the most common color
        flat = quantized.reshape(-1, 3)
        colors, counts = np.unique(flat, axis=0, return_counts=True)
        dominant_frac = float(counts.max()) / float(len(flat))
        return dominant_frac

    def _background_uniformity(self, image: np.ndarray) -> float:
        """Measure how uniform the background is.

        Samples corner regions (likely background) and measures their variance.
        AI images with synthetic backgrounds have very low corner variance.
        Returns a normalized score in [0, 1] where 0 = perfectly uniform, 1 = noisy.
        """
        img_f = image.astype(np.float32) / 255.0
        h, w = img_f.shape[:2]
        corner_size = min(h, w) // 8
        corners = [
            img_f[:corner_size, :corner_size],
            img_f[:corner_size, -corner_size:],
            img_f[-corner_size:, :corner_size],
            img_f[-corner_size:, -corner_size:],
        ]
        # Measure variance across all corners (low = uniform background = AI)
        corner_means = [np.mean(c, axis=(0, 1)) for c in corners]
        cross_corner_var = float(np.std(corner_means))
        # Also measure within-corner variance
        within_corner_vars = [float(np.mean(np.var(c, axis=(0, 1)))) for c in corners]
        mean_within_var = float(np.mean(within_corner_vars))
        # Combined: low variance = uniform = AI. Normalize to [0, 1] range.
        # Typical values: AI dark bg ~0.001, real photo ~0.05
        combined_var = cross_corner_var + mean_within_var
        return float(np.clip(combined_var, 0.0, 1.0))

    def analyze(self, image: np.ndarray) -> SignalResult:
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        palette_ratio = self._color_palette_size(image)
        edge_density = self._edge_density(image)
        smooth_ratio = self._smooth_ratio(image)
        channel_corr = self._channel_correlation(image)
        gradient_smooth = self._gradient_smoothness(image)
        dominant_color = self._dominant_color_concentration(image)
        bg_uniformity = self._background_uniformity(image)

        # Score each feature
        # 1. Low palette ratio -> AI (few unique colors = synthetic)
        palette_score = 1.0 / (1.0 + np.exp(-20.0 * (self.palette_ratio_threshold - palette_ratio)))

        # 2. High smooth ratio -> AI (too many smooth blocks)
        smooth_score = 1.0 / (1.0 + np.exp(-15.0 * (smooth_ratio - self.smooth_ratio_threshold)))

        # 3. High channel correlation -> AI (jointly synthesized channels)
        corr_score = 1.0 / (1.0 + np.exp(-30.0 * (channel_corr - self.channel_corr_threshold)))

        # 4. Edge density outside natural range -> AI
        lo, hi = self.edge_density_range
        if edge_density < lo:
            edge_score = 1.0 / (1.0 + np.exp(-50.0 * (lo - edge_density)))
        elif edge_density > hi:
            edge_score = 1.0 / (1.0 + np.exp(-30.0 * (edge_density - hi)))
        else:
            edge_score = 0.3  # neutral

        # 5. High gradient smoothness -> AI
        grad_score = 1.0 / (1.0 + np.exp(-15.0 * (gradient_smooth - 0.6)))

        # 6. High dominant color concentration -> AI
        # AI images with solid backgrounds have >35% of pixels in one color
        dominant_score = 1.0 / (1.0 + np.exp(-15.0 * (dominant_color - self.dominant_color_threshold)))

        # 7. Low background variance -> AI (synthetic uniform background)
        # bg_uniformity is normalized [0, 1]: low = uniform = AI
        # AI dark backgrounds: ~0.001-0.01, real photos: ~0.03-0.1+
        bg_score = 1.0 / (1.0 + np.exp(-80.0 * (0.02 - bg_uniformity)))

        # Weighted combination - dominant_color and bg_uniformity are key
        # for text/diagram AI images that other features miss
        combined = float(np.clip(
            0.15 * palette_score +
            0.10 * smooth_score +
            0.10 * corr_score +
            0.10 * edge_score +
            0.05 * grad_score +
            0.25 * dominant_score +
            0.25 * bg_score,
            0.0, 1.0
        ))

        # Content-awareness: if the image is text/diagram heavy (high edge density),
        # pixel-level semantic features are less reliable because text creates
        # natural-looking diversity. Reduce weight and pull toward neutral.
        is_text_heavy = edge_density > 0.20
        if is_text_heavy:
            # Blend toward neutral and reduce weight
            text_factor = float(np.clip((edge_density - 0.20) / 0.15, 0.0, 0.7))
            combined = float(0.5 * text_factor + combined * (1.0 - text_factor))
            weight = 0.75 * (1.0 - text_factor * 0.8)
        else:
            weight = 0.75

        return SignalResult(
            score=combined,
            weight=weight,
            features={
                "palette_ratio": float(palette_ratio),
                "edge_density": float(edge_density),
                "smooth_ratio": float(smooth_ratio),
                "channel_corr": float(channel_corr),
                "gradient_smoothness": float(gradient_smooth),
                "dominant_color_frac": float(dominant_color),
                "background_uniformity": float(bg_uniformity),
                "palette_score": float(palette_score),
                "smooth_score": float(smooth_score),
                "corr_score": float(corr_score),
                "edge_score": float(edge_score),
                "grad_score": float(grad_score),
                "dominant_score": float(dominant_score),
                "bg_score": float(bg_score),
            },
        )
