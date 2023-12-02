"""
adaptive_threshold.py

Rolling mean ± k * rolling std detector with robust statistics.

This is the simplest thing that actually works for most sensor
monitoring use cases. No training, no parameters to tune beyond k,
handles non-stationary signals reasonably well.

Uses MAD (Median Absolute Deviation) instead of standard deviation
for the rolling spread estimate — much more robust to outliers.
The problem with rolling std is that anomalies inflate it, which
raises the threshold right when you need it most. MAD doesn't do that.

Good first choice when you need something running quickly and don't
have labelled anomaly data to tune anything on.
"""

import numpy as np
from collections import deque
from .base import StreamDetector, AnomalyResult


class AdaptiveThreshold(StreamDetector):
    """
    Rolling adaptive threshold detector.

    Args:
        window: number of recent samples for rolling stats
        k: sensitivity multiplier (lower = more sensitive)
        use_mad: use MAD instead of std for spread (recommended)
        min_samples: minimum samples before flagging anomalies
    """

    def __init__(
        self,
        window: int = 100,
        k: float = 3.0,
        use_mad: bool = True,
        min_samples: int = 20,
    ):
        super().__init__(name="AdaptiveThreshold")
        self.window = window
        self.k = k
        self.use_mad = use_mad
        self.min_samples = min_samples
        self._buffer = deque(maxlen=window)

    @property
    def warmup_required(self) -> int:
        return self.min_samples

    def _rolling_stats(self) -> tuple[float, float]:
        """Returns (center, spread) for the current buffer."""
        arr = np.array(self._buffer)
        center = float(np.median(arr))
        if self.use_mad:
            spread = float(np.median(np.abs(arr - center))) * 1.4826  # scale to match std
        else:
            spread = float(np.std(arr))
        return center, max(spread, 1e-6)  # avoid division by zero

    def update(self, value: float) -> AnomalyResult:
        self._buffer.append(value)

        if len(self._buffer) < self.min_samples:
            result = AnomalyResult(is_anomaly=False, score=0.0, threshold=0.0)
            self._record(result)
            return result

        center, spread = self._rolling_stats()
        threshold = self.k * spread
        score = abs(value - center) / spread

        result = AnomalyResult(
            is_anomaly=score > self.k,
            score=score,
            threshold=self.k,
            metadata={"center": center, "spread": spread, "value": value},
        )
        self._record(result)
        return result

    def reset(self):
        self._buffer.clear()
        self._n_processed = 0
        self._n_anomalies = 0


class MultiStreamAdaptiveThreshold:
    """
    Runs one AdaptiveThreshold detector per stream.
    Useful for monitoring a fleet of sensors simultaneously.

    Usage:
        detector = MultiStreamAdaptiveThreshold(k=3.0, window=100)
        result = detector.update("rack_R07_inlet_temp", 24.5)
    """

    def __init__(self, **detector_kwargs):
        self._kwargs = detector_kwargs
        self._detectors: dict[str, AdaptiveThreshold] = {}

    def update(self, stream_id: str, value: float) -> AnomalyResult:
        if stream_id not in self._detectors:
            self._detectors[stream_id] = AdaptiveThreshold(**self._kwargs)
        return self._detectors[stream_id].update(value)

    def reset(self, stream_id: str = None):
        if stream_id:
            if stream_id in self._detectors:
                self._detectors[stream_id].reset()
        else:
            for d in self._detectors.values():
                d.reset()

    @property
    def n_streams(self) -> int:
        return len(self._detectors)

    def summary(self) -> dict:
        return {
            sid: {"n_processed": d._n_processed, "anomaly_rate": d.anomaly_rate}
            for sid, d in self._detectors.items()
        }
