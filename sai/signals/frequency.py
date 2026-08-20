"""Frequency-domain signal.

Synthetic generators (diffusion, GAN) leave periodic artifacts in the DCT
spectrum that real-camera captures do not. This signal measures two things:

1. High-frequency energy ratio: AI images often have anomalous spectral tails.
2. Cross-channel frequency cross-correlation: generators synthesize channels
   jointly, which produces a cross-channel spectral signature distinct from
   the per-channel Bayer-pattern demosaicing of a real camera.

The two features are combined with a logistic map calibrated on a held-out set.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import fft2, fftshift

from sai.signals.base import Signal, SignalResult


class FrequencySignal(Signal):
    name = "frequency"

    def __init__(
        self,
        hf_ratio_weight: float = 0.6,
        cross_corr_weight: float = 0.4,
        logistic_k: float = 8.0,
        logistic_x0: float = 0.45,
    ) -> None:
        self.hf_ratio_weight = hf_ratio_weight
        self.cross_corr_weight = cross_corr_weight
        self.logistic_k = logistic_k
        self.logistic_x0 = logistic_x0

    def _channel_spectrum(self, channel: np.ndarray) -> np.ndarray:
        # log-magnitude spectrum, centered
        spec = fftshift(fft2(channel.astype(np.float32)))
        mag = np.log1p(np.abs(spec))
        return mag

    def _hf_ratio(self, mag: np.ndarray) -> float:
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        max_r = np.hypot(cy, cx)
        hf_mask = r > 0.4 * max_r
        lf_mask = r <= 0.4 * max_r
        hf_energy = float(mag[hf_mask].sum())
        lf_energy = float(mag[lf_mask].sum()) + 1e-6
        return hf_energy / (hf_energy + lf_energy)

    def _cross_corr(self, spectra: list[np.ndarray]) -> float:
        # mean pairwise correlation of flattened spectra across channels
        flat = [s.flatten() for s in spectra]
        n = len(flat)
        if n < 2:
            return 0.0
        corrs = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = flat[i], flat[j]
                a_c = a - a.mean()
                b_c = b - b.mean()
                denom = (np.linalg.norm(a_c) * np.linalg.norm(b_c)) + 1e-9
                corrs.append(float(np.dot(a_c, b_c) / denom))
        return float(np.mean(corrs))

    def analyze(self, image: np.ndarray) -> SignalResult:
        if image.ndim != 3 or image.shape[2] < 2:
            # single-channel: duplicate to compute cross-corr meaningfully
            image = np.repeat(image[..., None], 3, axis=2) if image.ndim == 2 else image
        spectra = [self._channel_spectrum(image[:, :, c]) for c in range(image.shape[2])]
        hf_ratios = [self._hf_ratio(s) for s in spectra]
        hf_mean = float(np.mean(hf_ratios))
        xcorr = self._cross_corr(spectra)

        # Combined feature: AI images tend to have higher cross-corr and
        # shifted hf_ratio. Both are mapped to [0,1] then combined.
        hf_score = 1.0 / (1.0 + np.exp(-self.logistic_k * (hf_mean - self.logistic_x0)))
        # cross-corr for real images is typically < 0.5; for AI often > 0.7
        xc_score = 1.0 / (1.0 + np.exp(-12.0 * (xcorr - 0.65)))
        combined = self.hf_ratio_weight * hf_score + self.cross_corr_weight * xc_score
        combined = float(np.clip(combined, 0.0, 1.0))

        return SignalResult(
            score=combined,
            weight=0.7,
            features={
                "hf_ratio": hf_mean,
                "cross_corr": xcorr,
                "hf_score": float(hf_score),
                "xc_score": float(xc_score),
            },
        )
