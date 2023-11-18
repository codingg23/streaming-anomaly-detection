"""
cusum.py

CUSUM (Cumulative Sum) change point detector.

One of the oldest and most reliable algorithms for detecting
shifts in a signal's mean. Originally developed for industrial
quality control in the 1950s — still competitive today for
this type of problem.

The idea: accumulate deviations from a reference value.
When the cumulative sum exceeds a threshold, flag an anomaly.
Two-sided version catches both upward and downward shifts.

Reference value (mu) and slack (k) need to be set appropriately:
  - mu: expected mean of normal data
  - k: allowable slack, usually k = delta/2 where delta is the
       smallest shift you care about detecting

I added an adaptive warm-up mode that estimates mu and sigma
from the first N samples, which is useful when you don't know
the signal's baseline a priori.
"""

import numpy as np
from collections import deque
from .base import StreamDetector, AnomalyResult


class CUSUM(StreamDetector):
    """
    Two-sided CUSUM detector.

    Args:
        threshold: alarm threshold (H in the literature)
        slack: allowable deviation before accumulation starts (k)
        mu: expected mean — if None, estimated from warm-up samples
        sigma: expected std — if None, estimated from warm-up samples
        warmup_samples: number of samples to use for parameter estimation
    """

    def __init__(
        self,
        threshold: float = 5.0,
        slack: float = 1.0,
        mu: float = None,
        sigma: float = None,
        warmup_samples: int = 50,
    ):
        super().__init__(name="CUSUM")
        self.threshold = threshold
        self.slack = slack
        self._mu_init = mu
        self._sigma_init = sigma
        self._warmup_n = warmup_samples

        self._mu = mu
        self._sigma = sigma or 1.0
        self._cusum_pos = 0.0   # upward cumulative sum
        self._cusum_neg = 0.0   # downward cumulative sum
        self._warmup_buffer = deque(maxlen=warmup_samples) if mu is None else None

    @property
    def warmup_required(self) -> int:
        return self._warmup_n if self._mu_init is None else 0

    def _finish_warmup(self):
        data = np.array(self._warmup_buffer)
        self._mu = float(np.median(data))          # median more robust than mean
        self._sigma = float(np.std(data)) or 1.0
        # rescale slack in terms of sigma
        self._slack_scaled = self.slack * self._sigma
        self._warmup_buffer = None

    def update(self, value: float) -> AnomalyResult:
        # warm-up phase: collect samples to estimate parameters
        if self._warmup_buffer is not None:
            self._warmup_buffer.append(value)
            if len(self._warmup_buffer) >= self._warmup_n:
                self._finish_warmup()
            result = AnomalyResult(is_anomaly=False, score=0.0, threshold=self.threshold)
            self._record(result)
            return result

        if not hasattr(self, "_slack_scaled"):
            self._slack_scaled = self.slack * self._sigma

        # standardise the observation
        z = (value - self._mu) / self._sigma

        # update cumulative sums (two-sided)
        self._cusum_pos = max(0, self._cusum_pos + z - self._slack_scaled)
        self._cusum_neg = max(0, self._cusum_neg - z - self._slack_scaled)

        score = max(self._cusum_pos, self._cusum_neg)
        is_anomaly = score > self.threshold

        # reset after alarm (optional — some implementations keep accumulating)
        if is_anomaly:
            self._cusum_pos = 0.0
            self._cusum_neg = 0.0

        result = AnomalyResult(
            is_anomaly=is_anomaly,
            score=score,
            threshold=self.threshold,
            metadata={"cusum_pos": self._cusum_pos, "cusum_neg": self._cusum_neg, "z": z},
        )
        self._record(result)
        return result

    def reset(self):
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._n_processed = 0
        self._n_anomalies = 0
        if self._mu_init is None:
            self._mu = None
            self._warmup_buffer = deque(maxlen=self._warmup_n)


class AdaptiveCUSUM(CUSUM):
    """
    CUSUM with a slowly-updating reference to handle non-stationary signals.

    After warm-up, the reference mu is updated with an exponential
    moving average of non-anomalous samples. This lets the detector
    track slow drifts in the baseline without flagging them as anomalies.

    Not always the right call — if you WANT to detect slow drifts,
    use the regular CUSUM. This variant is for when you only care
    about sudden changes on top of a shifting baseline.
    """

    def __init__(self, ema_alpha: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.name = "AdaptiveCUSUM"
        self.ema_alpha = ema_alpha   # small = slow adaptation

    def update(self, value: float) -> AnomalyResult:
        result = super().update(value)
        # only update reference on normal samples
        if not result.is_anomaly and self._mu is not None:
            self._mu = (1 - self.ema_alpha) * self._mu + self.ema_alpha * value
        return result
