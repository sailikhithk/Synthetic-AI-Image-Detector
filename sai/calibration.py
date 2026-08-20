"""Calibration and uncertainty.

Two production-reliability primitives that are missing from most open-source
AI-image detectors:

1. Temperature scaling: post-hoc calibration that maps raw ensemble scores
   to well-calibrated probabilities. We fit a single temperature T on a
   held-out labeled set; calibrated_score = sigmoid(logit(raw) / T).

2. Uncertainty quantification: report epistemic (model) uncertainty via
   signal disagreement and aleatoric (data) uncertainty via signal weight
   entropy. When total uncertainty exceeds a threshold, the detector
   should refuse to commit and return verdict="inconclusive".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class CalibratedResult:
    calibrated_score: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    total_uncertainty: float
    verdict: str  # "ai", "real", or "inconclusive"


class TemperatureScaler:
    """Platt-style temperature scaling for binary ensemble scores."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = max(temperature, 1e-3)

    @staticmethod
    def _logit(p: float) -> float:
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        return float(np.log(p / (1 - p)))

    def transform(self, raw_score: float) -> float:
        z = self._logit(raw_score) / self.temperature
        return float(1.0 / (1.0 + np.exp(-z)))

    def fit(self, raw_scores: Iterable[float], labels: Iterable[int], max_iter: int = 200, lr: float = 0.1) -> float:
        """Gradient-descent fit of temperature on a held-out labeled set.

        Minimizes NLL of calibrated probabilities under binary labels.
        Returns the fitted temperature.
        """
        s = np.asarray(list(raw_scores), dtype=np.float64)
        y = np.asarray(list(labels), dtype=np.float64)
        # init from raw
        logits = np.log(np.clip(s, 1e-6, 1 - 1e-6) / np.clip(1 - s, 1e-6, 1 - 1e-6))
        T = 1.0
        for _ in range(max_iter):
            z = logits / T
            p = 1.0 / (1.0 + np.exp(-z))
            # gradient of NLL wrt T
            # NLL = -mean( y*log(p) + (1-y)*log(1-p) ), p = sigmoid(logits / T)
            # dNLL/dT = mean( (y - p) * logits / T^2 )
            grad = float(np.mean((y - p) * logits / (T * T)))
            T_new = T - lr * grad
            T_new = max(T_new, 1e-3)
            if abs(T_new - T) < 1e-6:
                T = T_new
                break
            T = T_new
        self.temperature = float(T)
        return self.temperature


def uncertainty(
    scores: np.ndarray,
    weights: np.ndarray,
    calibrated_score: float,
    refuse_threshold: float = 0.4,
) -> CalibratedResult:
    """Compute epistemic + aleatoric uncertainty from a set of signal results.

    epistemic: disagreement between signals (weighted variance of scores).
    aleatoric: entropy of the weight distribution (low confidence in any
               single signal -> high aleatoric uncertainty).
    """
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if scores.size == 0:
        return CalibratedResult(0.5, 1.0, 1.0, 1.0, "inconclusive")
    w = weights / (weights.sum() + 1e-9)
    weighted_mean = float(np.dot(w, scores))
    weighted_var = float(np.dot(w, (scores - weighted_mean) ** 2))
    epistemic = float(np.clip(weighted_var * 4.0, 0.0, 1.0))  # scale to [0,1]
    # entropy of weights, normalized
    p = w + 1e-12
    entropy = float(-np.sum(p * np.log(p)))
    max_entropy = float(np.log(len(w))) if len(w) > 1 else 1.0
    aleatoric = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    total = float(np.clip(0.6 * epistemic + 0.4 * aleatoric, 0.0, 1.0))

    if total >= refuse_threshold:
        verdict = "inconclusive"
    elif calibrated_score >= 0.5:
        verdict = "ai"
    else:
        verdict = "real"
    return CalibratedResult(
        calibrated_score=float(calibrated_score),
        epistemic_uncertainty=epistemic,
        aleatoric_uncertainty=aleatoric,
        total_uncertainty=total,
        verdict=verdict,
    )
