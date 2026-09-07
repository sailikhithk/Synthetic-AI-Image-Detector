"""Detector pipeline: ensemble of signals + calibration + uncertainty."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from sai.calibration import TemperatureScaler, uncertainty, CalibratedResult
from sai.signals import (
    Signal,
    SignalResult,
    FrequencySignal,
    ReconstructionSignal,
    NoiseResidualSignal,
    MetadataSignal,
    SemanticSignal,
    CrossImageConsistencySignal,
)


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
        if signals is not None:
            self.signals = signals
        else:
            self.signals = [
                FrequencySignal(),
                ReconstructionSignal(),
                NoiseResidualSignal(),
                SemanticSignal(),
            ]
        self.scaler = scaler or TemperatureScaler()
        self.refuse_threshold = refuse_threshold
        self._metadata_signal: Optional[MetadataSignal] = None
        self._cross_image_signal: Optional[CrossImageConsistencySignal] = None

    def detect(self, image: np.ndarray, file_path: Optional[str | Path] = None) -> DetectionResult:
        """Detect if a single image is AI-generated.

        Args:
            image: HxWxC uint8 RGB array.
            file_path: Optional path to the image file. If provided,
                       MetadataSignal will analyze EXIF/IPTC headers.
        """
        results = [s.analyze(image) for s in self.signals]

        # Add metadata signal if file path is provided
        if file_path is not None:
            if self._metadata_signal is None:
                self._metadata_signal = MetadataSignal()
            meta_result = self._metadata_signal.analyze_file(file_path)
            meta_result.features["name"] = "metadata"
            results.append(meta_result)

        # Set names on signal results
        for s, r in zip(self.signals, results[:len(self.signals)]):
            r.features["name"] = s.name

        scores = np.array([r.score for r in results])
        weights = np.array([r.weight for r in results])
        w = weights / (weights.sum() + 1e-9)
        raw = float(np.dot(w, scores))
        calibrated = self.scaler.transform(raw)
        cal: CalibratedResult = uncertainty(scores, weights, calibrated, self.refuse_threshold)
        features = {}
        for i, r in enumerate(results):
            if i < len(self.signals):
                r.features["name"] = self.signals[i].name
            elif "name" not in r.features:
                r.features["name"] = "metadata"
            features[r.features["name"]] = r.features
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

    def detect_batch(
        self,
        images: List[np.ndarray],
        file_paths: Optional[List[str | Path]] = None,
    ) -> List[DetectionResult]:
        """Detect if a batch of images is AI-generated.

        Uses CrossImageConsistencySignal to measure batch-level uniformity
        (AI-generated series have very consistent statistics across images).

        Args:
            images: List of HxWxC uint8 RGB arrays.
            file_paths: Optional list of file paths for metadata analysis.
        """
        if not images:
            return []

        # Run per-image signals
        per_image_results = []
        for i, img in enumerate(images):
            fp = file_paths[i] if file_paths else None
            results = [s.analyze(img) for s in self.signals]

            # Metadata signal
            if fp is not None:
                if self._metadata_signal is None:
                    self._metadata_signal = MetadataSignal()
                meta_result = self._metadata_signal.analyze_file(fp)
                meta_result.features["name"] = "metadata"
                results.append(meta_result)

            for j, r in enumerate(results[:len(self.signals)]):
                r.features["name"] = self.signals[j].name

            per_image_results.append(results)

        # Cross-image consistency signal
        if self._cross_image_signal is None:
            self._cross_image_signal = CrossImageConsistencySignal()
        cross_results = self._cross_image_signal.analyze_batch(images)
        for i, cr in enumerate(cross_results):
            cr.features["name"] = "cross_image"
            per_image_results[i].append(cr)

        # Build DetectionResult for each image
        all_results = []
        for results in per_image_results:
            scores = np.array([r.score for r in results])
            weights = np.array([r.weight for r in results])
            w = weights / (weights.sum() + 1e-9)
            raw = float(np.dot(w, scores))
            calibrated = self.scaler.transform(raw)
            cal: CalibratedResult = uncertainty(scores, weights, calibrated, self.refuse_threshold)
            features = {}
            for r in results:
                features[r.features.get("name", "?")] = r.features
            all_results.append(DetectionResult(
                raw_score=raw,
                calibrated_score=cal.calibrated_score,
                verdict=cal.verdict,
                epistemic_uncertainty=cal.epistemic_uncertainty,
                aleatoric_uncertainty=cal.aleatoric_uncertainty,
                total_uncertainty=cal.total_uncertainty,
                signal_results=results,
                features=features,
            ))
        return all_results

    def fit_calibration(self, raw_scores: List[float], labels: List[int]) -> float:
        return self.scaler.fit(raw_scores, labels)
