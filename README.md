# streaming-anomaly-detection

Online anomaly detection algorithms for high-frequency time-series data. Designed to run continuously on streaming sensor data with low latency and no retraining.

Built this because most anomaly detection libraries assume you have all your data upfront. In practice, sensor streams never stop — you need algorithms that update their model incrementally as new data arrives and flag anomalies in real time.

## Overview

Four algorithms implemented, each with different trade-offs:

| Algorithm | Latency | Memory | Works well for |
|---|---|---|---|
| CUSUM | < 1ms | O(1) | Gradual drifts, mean shifts |
| Streaming Isolation Forest | ~5ms | O(window) | Multivariate point anomalies |
| LSTM Autoencoder | ~20ms | O(model) | Contextual / seasonal anomalies |
| Adaptive Threshold | < 1ms | O(window) | Unknown distributions, fast setup |

The right choice depends on your use case. For most sensor monitoring I'd start with Adaptive Threshold to get something running fast, then layer in the LSTM autoencoder once you have enough history.

## Features

- All detectors implement a common `StreamDetector` interface — easy to swap
- Windowed statistics updated incrementally (no recomputation from scratch)
- Configurable sensitivity and alarm thresholds
- Multi-stream support: run one detector per signal, or use multivariate detectors
- Evaluation tools: inject synthetic anomalies and measure precision/recall
- Export alerts as structured JSON events

## Tech Stack

- Python 3.11
- NumPy / SciPy
- PyTorch (LSTM autoencoder only)
- `river` library for some streaming stats primitives

## How to Run

```bash
git clone https://github.com/codingg23/streaming-anomaly-detection
cd streaming-anomaly-detection
pip install -r requirements.txt

# run on a CSV of time-series data
python detect.py --data ./data/example_stream.csv --detector cusum --sensitivity 3.0

# evaluate with injected anomalies
python evaluate.py --data ./data/example_stream.csv --detector lstm_ae --inject-rate 0.02

# run all detectors and compare
python benchmark.py --data ./data/example_stream.csv
```

## Algorithm Notes

**CUSUM** is the workhorse — almost no compute, catches mean shifts within a few samples. The trick is setting the reference value and slack parameter correctly, which depends on your expected change magnitude. I added an adaptive variant that estimates these from a warm-up window.

**Streaming Isolation Forest** extends the original iForest to a sliding window. Anomaly score is based on average path length in random trees, same as the batch version. The main challenge is efficiently updating trees as old points leave the window — current implementation rebuilds periodically (a TODO to do this properly incrementally).

**LSTM Autoencoder** is the most powerful but slowest. Train it on normal data, flag samples where reconstruction error exceeds a threshold. Good at catching anomalies that are only anomalous in context (e.g. temperature reading that's fine in isolation but wrong given recent history).

**Adaptive Threshold** is the simplest thing that actually works: rolling mean ± k * rolling std, with the std estimated robustly using MAD. Handles non-stationary signals surprisingly well.

## Results / Learnings

On synthetic benchmarks with injected point and contextual anomalies:

| Algorithm | Precision | Recall | F1 |
|---|---|---|---|
| Adaptive Threshold | 0.71 | 0.83 | 0.77 |
| CUSUM | 0.78 | 0.76 | 0.77 |
| Streaming iForest | 0.82 | 0.74 | 0.78 |
| LSTM Autoencoder | 0.86 | 0.81 | 0.83 |

The LSTM wins on F1 but the latency difference is real. For anything
latency-sensitive (< 5ms budget), CUSUM or Adaptive Threshold is the answer.

Main lesson: precision matters more than recall for alerting systems.
False positives kill operator trust faster than missed anomalies.

## TODO

- [ ] Proper incremental iForest (not periodic rebuild)
- [ ] Multivariate CUSUM (currently univariate only)
- [ ] Kafka/Flink integration for distributed streams
- [ ] Anomaly correlation across multiple streams
