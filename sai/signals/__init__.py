"""Detection signals. Each signal produces a score in [0, 1] and a confidence weight."""

from sai.signals.base import Signal, SignalResult
from sai.signals.frequency import FrequencySignal
from sai.signals.reconstruction import ReconstructionSignal
from sai.signals.noise import NoiseResidualSignal

__all__ = [
    "Signal",
    "SignalResult",
    "FrequencySignal",
    "ReconstructionSignal",
    "NoiseResidualSignal",
]
