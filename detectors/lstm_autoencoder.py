"""
lstm_autoencoder.py

LSTM autoencoder for contextual anomaly detection.

Train on normal data, flag samples where reconstruction
error exceeds a calibrated threshold.

Why autoencoders work for anomaly detection:
The model learns to compress and reconstruct normal patterns.
When it sees something anomalous, it hasn't learned to reconstruct
it well, so the error is high. Simple in principle, works well
in practice for time-series with strong seasonal/contextual patterns.

The key challenge: setting the threshold. I use a calibration
set (a holdout from normal data) to set the threshold at a
target false positive rate. This is much more principled than
just picking a multiplier.

Note: this is the only detector here that requires training.
Use AdaptiveThreshold or CUSUM if you need zero-config.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from collections import deque
from .base import StreamDetector, AnomalyResult


@dataclass
class LSTMAEConfig:
    input_size: int = 1
    hidden_size: int = 64
    num_layers: int = 2
    window_size: int = 50        # context window for reconstruction
    latent_size: int = 16        # bottleneck size
    dropout: float = 0.1


class LSTMEncoder(nn.Module):
    def __init__(self, config: LSTMAEConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            config.input_size, config.hidden_size,
            config.num_layers, batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(config.hidden_size, config.latent_size)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class LSTMDecoder(nn.Module):
    def __init__(self, config: LSTMAEConfig):
        super().__init__()
        self.fc = nn.Linear(config.latent_size, config.hidden_size)
        self.lstm = nn.LSTM(
            config.hidden_size, config.hidden_size,
            config.num_layers, batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0
        )
        self.out = nn.Linear(config.hidden_size, config.input_size)
        self.window_size = config.window_size

    def forward(self, z, seq_len: int):
        # expand latent vector into sequence
        h0 = self.fc(z).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        inp = h0[-1].unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.lstm(inp, (h0, c0))
        return self.out(out)


class LSTMAutoencoder(nn.Module):
    def __init__(self, config: LSTMAEConfig):
        super().__init__()
        self.encoder = LSTMEncoder(config)
        self.decoder = LSTMDecoder(config)
        self.config = config

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z, x.size(1))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x_hat = self.forward(x)
            return torch.mean((x - x_hat) ** 2, dim=(1, 2))


class LSTMAutoencoderDetector(StreamDetector):
    """
    Online anomaly detector using a trained LSTM autoencoder.

    Maintains a rolling window of recent samples and computes
    reconstruction error on each new sample.

    Requires a trained model. Use train() to train from scratch,
    or load_checkpoint() to load a pre-trained model.
    """

    def __init__(self, config: Optional[LSTMAEConfig] = None, threshold: Optional[float] = None):
        super().__init__(name="LSTMAutoencoder")
        self.config = config or LSTMAEConfig()
        self.model = LSTMAutoencoder(self.config)
        self._threshold = threshold
        self._buffer = deque(maxlen=self.config.window_size)
        self._device = "cpu"

    @property
    def warmup_required(self) -> int:
        return self.config.window_size

    def train(self, normal_data: np.ndarray, epochs: int = 30, lr: float = 1e-3,
              val_frac: float = 0.1) -> list[float]:
        """
        Train the autoencoder on normal (anomaly-free) data.

        normal_data: 1D array of normal observations
        Returns: training loss history
        """
        from torch.utils.data import DataLoader, TensorDataset

        windows = self._make_windows(normal_data)
        n_val = max(1, int(len(windows) * val_frac))
        X_train = torch.from_numpy(windows[:-n_val]).float().unsqueeze(-1)
        X_val = torch.from_numpy(windows[-n_val:]).float().unsqueeze(-1)

        loader = DataLoader(TensorDataset(X_train), batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                out = self.model(batch)
                loss = criterion(out, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))

        # calibrate threshold on val set at 99th percentile
        self.model.eval()
        val_errors = self.model.reconstruction_error(X_val).numpy()
        self._threshold = float(np.percentile(val_errors, 99))
        return losses

    def _make_windows(self, data: np.ndarray) -> np.ndarray:
        w = self.config.window_size
        return np.array([data[i:i+w] for i in range(len(data) - w + 1)])

    def update(self, value: float) -> AnomalyResult:
        self._buffer.append(value)

        if len(self._buffer) < self.config.window_size:
            result = AnomalyResult(is_anomaly=False, score=0.0, threshold=self._threshold or 0.0)
            self._record(result)
            return result

        if self._threshold is None:
            raise RuntimeError("Threshold not set. Call train() or set threshold manually.")

        window = np.array(self._buffer, dtype=np.float32)
        # normalise the window
        mu, std = window.mean(), window.std() + 1e-6
        window_norm = (window - mu) / std

        x = torch.from_numpy(window_norm).float().unsqueeze(0).unsqueeze(-1)
        self.model.eval()
        score = float(self.model.reconstruction_error(x).item())

        result = AnomalyResult(
            is_anomaly=score > self._threshold,
            score=score,
            threshold=self._threshold,
        )
        self._record(result)
        return result

    def reset(self):
        self._buffer.clear()
        self._n_processed = 0
        self._n_anomalies = 0

    def save(self, path: str):
        torch.save({"model": self.model.state_dict(), "threshold": self._threshold,
                    "config": self.config}, path)

    @classmethod
    def load(cls, path: str) -> "LSTMAutoencoderDetector":
        ckpt = torch.load(path, map_location="cpu")
        d = cls(config=ckpt["config"], threshold=ckpt["threshold"])
        d.model.load_state_dict(ckpt["model"])
        return d
