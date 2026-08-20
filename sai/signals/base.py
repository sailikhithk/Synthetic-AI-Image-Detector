"""Signal base class. Every detector signal conforms to this interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SignalResult:
    """Output of a single signal.

    score: probability the image is AI-generated, in [0, 1].
    weight: how much the ensemble should trust this signal, in [0, 1].
    features: optional dict of diagnostic features for logging/interpretability.
    """

    score: float
    weight: float
    features: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0,1], got {self.score}")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight must be in [0,1], got {self.weight}")


class Signal(ABC):
    """A detector signal. Stateless across calls; carry config on the instance."""

    name: str = "base"

    @abstractmethod
    def analyze(self, image: np.ndarray) -> SignalResult:
        """Run the signal on an HxWxC uint8 RGB array. Return a SignalResult."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Signal {self.name}>"
