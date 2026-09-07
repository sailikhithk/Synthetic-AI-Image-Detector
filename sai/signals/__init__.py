"""Detection signals. Each signal produces a score in [0, 1] and a confidence weight."""

from sai.signals.base import Signal, SignalResult
from sai.signals.frequency import FrequencySignal
from sai.signals.reconstruction import ReconstructionSignal
from sai.signals.noise import NoiseResidualSignal
from sai.signals.metadata import MetadataSignal
from sai.signals.semantic import SemanticSignal
from sai.signals.cross_image import CrossImageConsistencySignal

__all__ = [
    "Signal",
    "SignalResult",
    "FrequencySignal",
    "ReconstructionSignal",
    "NoiseResidualSignal",
    "MetadataSignal",
    "SemanticSignal",
    "CrossImageConsistencySignal",
]
