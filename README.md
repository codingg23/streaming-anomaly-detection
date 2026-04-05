# streaming-anomaly-detection

Online anomaly detection algorithms for high-frequency time-series data. Designed to run continuously on streaming sensor data with low latency and no retraining.

Built this because most anomaly detection libraries assume you have all your data upfront. In practice, sensor streams never stop - you need algorithms that update their model incrementally as data arrives and flag anomalies in real time.

## Overview

Four algorithms, each with different trade-offs:

| Algorithm | Latency | Memory | Works well for |
|---|---|---|---|
| CUSUM | < 1ms | O(1) | Gradual drifts, mean shifts |
| Streaming Isolation Forest | ~5ms | O(window) | Multivariate point anomalies |
| LSTM Autoencoder | ~20ms | O(model) | Contextual / seasonal anomalies |
| Adaptive Threshold | < 1ms | O(window) | Unknown distributions, fast setup |

For most sensor monitoring I'd start with Adaptive Threshold to get something running quickly, then layer in the LSTM autoencoder once you have enough data to train on.

## Features

- All detectors implement a common `StreamDetector` interface - easy to swap between them
- Windowed statistics updated incrementally (no recomputation from scratch)
- Configurable sensitivity and alarm thresholds
- Multi-stream support: run one detector per signal or use multivariate detectors
- Evaluation tools: inject synthetic anomalies and measure precision/recall
- Alert export as structured JSON events

## Tech Stack

- Python 3.11
- NumPy / SciPy
- PyTorch (LSTM autoencoder)
- `river` library for streaming stats primitives

## How to Run

```bash
git clone https://github.com/codingg23/streaming-anomaly-detection
cd streaming-anomaly-detection
pip install -r requirements.txt

# run on a CSV of time-series data
python detect.py --data ./data/example_stream.csv --detector cusum --sensitivity 3.0

# evaluate with injected anomalies
python evaluate.py --data ./data/example_stream.csv --detector lstm_ae --inject-rate 0.02

# compare all detectors on the same data
python benchmark.py --data ./data/example_stream.csv
```

## Algorithm Notes

**CUSUM** is the workhorse - almost no compute, catches mean shifts within a few samples. The main challenge is setting the reference value and slack parameter correctly. Added an adaptive variant that estimates these from a warm-up window when you don't know the baseline.

**Streaming Isolation Forest** extends the original iForest to a sliding window. Anomaly score is based on average path length in random trees, same as the batch version. Current implementation rebuilds trees periodically - a proper incremental update is a TODO.

**LSTM Autoencoder** is the most accurate but slowest. Train it on normal data, flag samples where reconstruction error exceeds a calibrated threshold. Catches anomalies that are only anomalous in context (e.g. a temperature reading that's fine in isolation but wrong given the last hour of history).

**Adaptive Threshold** is the simplest thing that actually works: rolling median +/- k times rolling spread, with spread estimated using MAD instead of std. MAD doesn't get inflated by the anomalies themselves, which is the key advantage over a simple rolling z-score.

## Results / Learnings

On synthetic benchmarks with injected point and contextual anomalies:

| Algorithm | Precision | Recall | F1 |
|---|---|---|---|
| Adaptive Threshold | 0.71 | 0.83 | 0.77 |
| CUSUM | 0.78 | 0.76 | 0.77 |
| Streaming iForest | 0.82 | 0.74 | 0.78 |
| LSTM Autoencoder | 0.86 | 0.81 | 0.83 |

LSTM wins on F1 but the latency difference is real. For anything where you need sub-5ms response, CUSUM or Adaptive Threshold is the right call.

Main lesson from using this on real sensor streams: precision matters more than recall for alerting. False positives destroy operator trust faster than missed anomalies.

## TODO

- [ ] Proper incremental iForest update (not periodic rebuild)
- [ ] Multivariate CUSUM
- [ ] Kafka/Flink integration for distributed streams
- [ ] Anomaly correlation across multiple streams
