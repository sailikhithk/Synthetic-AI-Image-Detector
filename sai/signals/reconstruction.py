"""Reconstruction-error signal.

Diffusion and GAN images often reconstruct cleanly under a generic
natural-image denoiser because they live on the generator's manifold.
Real photographs contain sensor noise and demosaicing artifacts that
a generic denoiser leaves as residual.

This signal uses a wavelet-based denoiser (no torch dependency) as the
reconstructor and measures the residual energy. Low residual energy
relative to the image energy suggests the image is on a smooth
synthetic manifold.

We also measure residual high-frequency structure: AI images tend to
have a residual whose energy is concentrated in low frequencies (the
generator's typical failure modes), while real images have residual
spread across high frequencies (sensor noise).
"""

from __future__ import annotations

import numpy as np
from skimage.restoration import denoise_wavelet, estimate_sigma

from sai.signals.base import Signal, SignalResult


class ReconstructionSignal(Signal):
    name = "reconstruction"

    def __init__(
        self,
        residual_threshold: float = 0.015,
        logistic_k: float = 180.0,
    ) -> None:
        self.residual_threshold = residual_threshold
        self.logistic_k = logistic_k

    def analyze(self, image: np.ndarray) -> SignalResult:
        img_f = image.astype(np.float32) / 255.0
        if img_f.ndim == 2:
            img_f = np.stack([img_f] * 3, axis=-1)

        try:
            sigma_est = float(estimate_sigma(img_f, channel_axis=-1, average_sigmas=True))
        except Exception:
            sigma_est = 0.05

        denoised = denoise_wavelet(
            img_f,
            channel_axis=-1,
            convert2ycbcr=True,
            mode="soft",
            sigma=sigma_est,
            rescale_sigma=True,
        )
        residual = img_f - denoised
        residual_energy = float(np.mean(residual ** 2))
        # Spectral concentration of residual: low-freq fraction
        per_channel_lf = []
        for c in range(residual.shape[2]):
            ch = residual[:, :, c]
            spec = np.abs(np.fft.fftshift(np.fft.fft2(ch)))
            h, w = spec.shape
            cy, cx = h // 2, w // 2
            yy, xx = np.ogrid[:h, :w]
            r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            max_r = np.hypot(cy, cx)
            lf = float(spec[r <= 0.2 * max_r].sum())
            total = float(spec.sum()) + 1e-9
            per_channel_lf.append(lf / total)
        lf_frac = float(np.mean(per_channel_lf))

        # Low residual energy -> more likely AI.
        low_res_score = 1.0 / (1.0 + np.exp(-self.logistic_k * (self.residual_threshold - residual_energy)))
        # High lf_frac of residual -> more likely AI (generator failure mode).
        lf_score = 1.0 / (1.0 + np.exp(-20.0 * (lf_frac - 0.25)))
        combined = float(np.clip(0.6 * low_res_score + 0.4 * lf_score, 0.0, 1.0))

        return SignalResult(
            score=combined,
            weight=0.65,
            features={
                "residual_energy": residual_energy,
                "sigma_est": sigma_est,
                "residual_lf_frac": lf_frac,
                "low_res_score": float(low_res_score),
                "lf_score": float(lf_score),
            },
        )
