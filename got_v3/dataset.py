from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CFDataset(Dataset):
    """
    Loads polynomial CF coefficient arrays and labels.

    Required files in data_dir:
        an_coeffs.npy   (N, A)   numerator polynomial coefficients
        bn_coeffs.npy   (N, B)   denominator polynomial coefficients
        labels.npy      (N,)     log10|s·v - zeta3| targets

    Optional files (zero/fallback if absent):
        conv_rate.npy   (N,)     convergence rate proxy (z-scored)
        plateau.npy     (N,)     binary plateau indicator
        components.npy  (N,)     integer mixture labels  (k-means computed if missing)
    """

    def __init__(self, data_dir: str, n_components: int = 20):
        self.data_dir = Path(data_dir)
        self.n_components = n_components

        self.an_raw = np.load(self.data_dir / "an_coeffs.npy").astype(np.float32)
        self.bn_raw = np.load(self.data_dir / "bn_coeffs.npy").astype(np.float32)
        self.y_delta = np.load(self.data_dir / "labels.npy").astype(np.float32)
        self.N = len(self.y_delta)

        self.conv_rate = self._load_optional("conv_rate.npy", np.zeros(self.N, np.float32))
        self.plateau   = self._load_optional("plateau.npy",   np.zeros(self.N, np.float32))

        # Per-dimension max-abs normalisation
        self.an_scale = np.maximum(np.abs(self.an_raw).max(0), 1.0)
        self.bn_scale = np.maximum(np.abs(self.bn_raw).max(0), 1.0)
        self.an = (self.an_raw / self.an_scale).astype(np.float32)
        self.bn = (self.bn_raw / self.bn_scale).astype(np.float32)

        # Concatenated coefficient vector used by neighbourhood/operator losses
        self.coeff = np.concatenate([self.an, self.bn], axis=1).astype(np.float32)

        # Clamp regression target to avoid exploding gradients
        self.y_delta = np.clip(self.y_delta, -14.0, 3.0).astype(np.float32)

        # Z-score convergence rate proxy
        self.conv_rate = self._zscore(self.conv_rate.astype(np.float32))

        # Mixture component labels
        comp_path = self.data_dir / "components.npy"
        if comp_path.exists():
            self.components = np.load(comp_path).astype(np.int64)
        else:
            self.components = self._kmeans_components()

    # ------------------------------------------------------------------
    def _load_optional(self, filename: str, default: np.ndarray) -> np.ndarray:
        path = self.data_dir / filename
        return np.load(path).astype(default.dtype) if path.exists() else default.copy()

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        sd = float(x.std())
        return (x - float(x.mean())) / sd if sd > 1e-8 else np.zeros_like(x)

    def _kmeans_components(self) -> np.ndarray:
        from sklearn.cluster import MiniBatchKMeans
        print(f"[dataset] k-means K={self.n_components} on coefficients …")
        km = MiniBatchKMeans(
            n_clusters=self.n_components, random_state=42,
            n_init=3, batch_size=4096,
        )
        return km.fit_predict(self.coeff).astype(np.int64)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx):
        return {
            "an":        torch.tensor(self.an[idx],          dtype=torch.float32),
            "bn":        torch.tensor(self.bn[idx],          dtype=torch.float32),
            "coeff":     torch.tensor(self.coeff[idx],       dtype=torch.float32),
            "delta":     torch.tensor(self.y_delta[idx],     dtype=torch.float32),
            "component": torch.tensor(self.components[idx],  dtype=torch.long),
            "conv_rate": torch.tensor(self.conv_rate[idx],   dtype=torch.float32),
            "plateau":   torch.tensor(self.plateau[idx],     dtype=torch.float32),
        }
