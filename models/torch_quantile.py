"""Optional PyTorch multi-quantile regressor for the pooled tabular ensemble."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _QuantileNetwork(nn.Module):
    def __init__(self, n_features: int, n_quantiles: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, n_quantiles),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class TorchQuantileRegressor:
    """Small CPU network trained jointly with pinball and crossing losses."""

    def __init__(self, quantiles, *, epochs: int = 60, random_state: int = 42):
        self.quantiles = np.asarray(quantiles, dtype=np.float32)
        self.epochs = int(epochs)
        self.random_state = int(random_state)
        self.network = None
        self.x_mean = None
        self.x_scale = None
        self.y_mean = 0.0
        self.y_scale = 1.0

    def fit(self, x, y):
        torch.manual_seed(self.random_state)
        torch.set_num_threads(1)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.x_mean = x.mean(axis=0)
        self.x_scale = x.std(axis=0)
        self.x_scale[self.x_scale < 1e-6] = 1.0
        self.y_mean = float(y.mean())
        self.y_scale = max(float(y.std()), 1.0)
        normalized_x = (x - self.x_mean) / self.x_scale
        normalized_y = (y - self.y_mean) / self.y_scale
        dataset = TensorDataset(torch.from_numpy(normalized_x), torch.from_numpy(normalized_y))
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(dataset, batch_size=min(128, len(dataset)), shuffle=True, generator=generator)
        self.network = _QuantileNetwork(x.shape[1], len(self.quantiles))
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=2e-3, weight_decay=1e-4)
        levels = torch.tensor(self.quantiles).reshape(1, -1)
        self.network.train()
        for _ in range(self.epochs):
            for features, targets in loader:
                optimizer.zero_grad()
                predictions = self.network(features)
                errors = targets - predictions
                pinball = torch.maximum(levels * errors, (levels - 1.0) * errors).mean()
                crossing = torch.relu(predictions[:, :-1] - predictions[:, 1:]).mean()
                loss = pinball + 0.2 * crossing
                loss.backward()
                optimizer.step()
        return self

    def predict(self, x):
        if self.network is None:
            raise RuntimeError("TorchQuantileRegressor has not been fit")
        normalized = (np.asarray(x, dtype=np.float32) - self.x_mean) / self.x_scale
        self.network.eval()
        with torch.inference_mode():
            output = self.network(torch.from_numpy(normalized)).cpu().numpy()
        output = output * self.y_scale + self.y_mean
        return np.maximum.accumulate(output, axis=1)


__all__ = ["TorchQuantileRegressor"]
