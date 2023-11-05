"""
base.py

Abstract base class for all streaming anomaly detectors.

All detectors implement the same interface so they're
trivially swappable. The key method is update(value) which
processes one sample at a time and returns an AnomalyResult.

Design decision: update() is synchronous and processes one
sample at a time, not batches. This keeps the interface
simple and forces each algorithm to be genuinely online.
If you need batch processing, just loop.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class AnomalyResult:
    is_anomaly: bool
    score: float              # raw anomaly score (higher = more anomalous)
    threshold: float          # threshold at this timestep
    timestamp: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class StreamDetector(ABC):
    """
    Base class for online anomaly detectors.

    Subclasses must implement:
      - update(value) -> AnomalyResult
      - reset() -> None

    Optionally override:
      - warmup_required -> int  (samples needed before detection starts)
      - is_ready -> bool
    """

    def __init__(self, name: str):
        self.name = name
        self._n_processed = 0
        self._n_anomalies = 0

    @abstractmethod
    def update(self, value: float) -> AnomalyResult:
        """
        Process one new sample. Returns an AnomalyResult.
        Called once per incoming data point.
        """
        ...

    @abstractmethod
    def reset(self):
        """Reset detector state. Call when stream is restarted."""
        ...

    @property
    def warmup_required(self) -> int:
        """Number of samples needed before the detector is reliable."""
        return 0

    @property
    def is_ready(self) -> bool:
        return self._n_processed >= self.warmup_required

    @property
    def anomaly_rate(self) -> float:
        if self._n_processed == 0:
            return 0.0
        return self._n_anomalies / self._n_processed

    def _record(self, result: AnomalyResult):
        self._n_processed += 1
        if result.is_anomaly:
            self._n_anomalies += 1

    def __repr__(self):
        return f"{self.__class__.__name__}(processed={self._n_processed}, anomaly_rate={self.anomaly_rate:.3f})"
