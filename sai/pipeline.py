"""Detector pipeline: ensemble of signals + calibration + uncertainty."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from sai.calibration import TemperatureScaler, uncertainty, CalibratedResult
from sai.signals import Signal, SignalResult, FrequencySignal, ReconstructionSignal, NoiseResidualSignal


@dataclass
class DetectionResult:
    """Full result returned by DetectorPipeline.detect()."""

    raw_score: float
    calibrated_score: float
    verdict: str
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    total_uncertainty: float
    signal_results: List[SignalResult] = field(default_factory=list)
    features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "raw_score": self.raw_score,
            "calibrated_score": self.calibrated_score,
            "verdict": self.verdict,
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "aleatoric_uncertainty": self.aleatoric_uncertainty,
            "total_uncertainty": self.total_uncertainty,
            "signals": [
                {"name": r.features.get("name", "?"), "score": r.score, "weight": r.weight, "features": r.features}
                for r in self.signal_results
            ],
        }


class DetectorPipeline:
    """Ensemble detector with calibration and uncertainty."""

    def __init__(
        self,
        signals: Optional[List[Signal]] = None,
        scaler: Optional[TemperatureScaler] = None,
        refuse_threshold: float = 0.4,
    ) -> None:
        self.signals = signals or [FrequencySignal(), ReconstructionSignal(), NoiseResidualSignal()]
        self.scaler = scaler or TemperatureScaler()
        self.refuse_threshold = refuse_threshold

    def detect(self, image: np.ndarray) -> DetectionResult:
        results = [s.analyze(image) for s in self.signals]
        scores = np.array([r.score for r in results])
        weights = np.array([r.weight for r in results])
        w = weights / (weights.sum() + 1e-9)
        raw = float(np.dot(w, scores))
        calibrated = self.scaler.transform(raw)
        cal: CalibratedResult = uncertainty(scores, weights, calibrated, self.refuse_threshold)
        features = {}
        for s, r in zip(self.signals, results):
            r.features["name"] = s.name
            features[s.name] = r.features
        return DetectionResult(
            raw_score=raw,
            calibrated_score=cal.calibrated_score,
            verdict=cal.verdict,
            epistemic_uncertainty=cal.epistemic_uncertainty,
            aleatoric_uncertainty=cal.aleatoric_uncertainty,
            total_uncertainty=cal.total_uncertainty,
            signal_results=results,
            features=features,
        )

    def fit_calibration(self, raw_scores: List[float], labels: List[int]) -> float:
        return self.scaler.fit(raw_scores, labels)
