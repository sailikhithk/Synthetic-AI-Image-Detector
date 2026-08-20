"""SAI: Synthetic AI Image detector.

A multi-signal detection pipeline with calibration, uncertainty quantification,
and a cross-generator evaluation harness aimed at production reliability for
journalists, fact-checkers, and national-security analysts.
"""

from sai.pipeline import DetectorPipeline, DetectionResult

__version__ = "0.1.0"
__all__ = ["DetectorPipeline", "DetectionResult"]
