"""MNIST data utilities for the 8×8 temporal delay network.

Provides:
  load_mnist_8x8()   — download MNIST and resize to 8×8, returning numpy arrays
  encode_dataset()   — batch-encode pixel arrays to arrival-time tensors

The 8×8 resolution is chosen to match the temporal network's delay budget:
  - 64 pixel inputs + 1 bias = 65-dimensional arrival-time vector
  - Pixel values in [0,1] are mapped to arrival times via encode_time()
  - Bright pixels (near 1) fire early (T ≈ 0 ns)
  - Dark pixels (near 0) fire late (T ≈ T_inactive = 150 ns)

Architecture: MemristorNet(n_inputs=64, hidden_sizes=[32], n_outputs=10)
  Layer 0: DelayLayer(65, 32)  → 65×32 = 2080 signed branch pairs
  Layer 1: DelayLayer(33, 10)  → 33×10 = 330 signed branch pairs
  Total: 2410 pairs × 2 = 4820 individual delay cells
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as tvt

_DATA_DIR = Path(__file__).parent.parent / "data"
_IMG_SIZE = 8


def load_mnist_8x8(
    data_dir: str | Path | None = None,
    max_train: int | None = None,
    max_test: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download (or load cached) MNIST and return 8×8 normalised arrays.

    Parameters
    ----------
    data_dir:
        Directory for torchvision to cache the raw MNIST files.
        Defaults to <repo_root>/data.
    max_train, max_test:
        If set, subsample to this many examples (stratified by class).
        Useful for fast experiments; None means use full dataset.
    seed:
        Random seed for subsampling.

    Returns
    -------
    X_train : (N_train, 64) float32  — pixels in [0, 1]
    y_train : (N_train,)   int64
    X_test  : (N_test, 64) float32
    y_test  : (N_test,)    int64
    """
    root = Path(data_dir) if data_dir is not None else _DATA_DIR
    root.mkdir(parents=True, exist_ok=True)

    transform = tvt.Compose([
        tvt.Resize((_IMG_SIZE, _IMG_SIZE), interpolation=tvt.InterpolationMode.BILINEAR),
        tvt.ToTensor(),  # → [0, 1] float32
    ])

    train_ds = torchvision.datasets.MNIST(
        root=str(root), train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.MNIST(
        root=str(root), train=False, download=True, transform=transform
    )

    def _to_numpy(ds: torchvision.datasets.MNIST) -> tuple[np.ndarray, np.ndarray]:
        X = ds.data  # (N, 28, 28) uint8 — load raw and transform manually for speed
        # Use torchvision resize on the whole tensor at once
        X_f = X.float().unsqueeze(1) / 255.0  # (N, 1, 28, 28)
        X_small = torch.nn.functional.interpolate(
            X_f, size=(_IMG_SIZE, _IMG_SIZE), mode="bilinear", align_corners=False
        )  # (N, 1, 8, 8)
        X_flat = X_small.squeeze(1).reshape(len(X_f), -1).numpy()  # (N, 64)
        y = ds.targets.numpy()
        return X_flat.astype(np.float32), y.astype(np.int64)

    X_train, y_train = _to_numpy(train_ds)
    X_test, y_test = _to_numpy(test_ds)

    if max_train is not None:
        X_train, y_train = _subsample(X_train, y_train, max_train, seed)
    if max_test is not None:
        X_test, y_test = _subsample(X_test, y_test, max_test, seed + 1)

    return X_train, y_train, X_test, y_test


def _subsample(
    X: np.ndarray, y: np.ndarray, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified subsample: select n//10 examples per class."""
    rng = np.random.default_rng(seed)
    per_class = max(1, n // 10)
    indices = []
    for c in range(10):
        mask = np.where(y == c)[0]
        chosen = rng.choice(mask, size=min(per_class, len(mask)), replace=False)
        indices.append(chosen)
    idx = np.concatenate(indices)
    idx = rng.permutation(idx)
    return X[idx], y[idx]


def encode_dataset(
    X: np.ndarray,
    net_encode_fn: object,
) -> torch.Tensor:
    """Batch-encode pixel array to arrival times using net.encode_time().

    Parameters
    ----------
    X:
        (N, 64) float32 pixel array.
    net_encode_fn:
        MemristorNet.encode_time bound method.

    Returns
    -------
    (N, 65) float32 tensor — arrival times including bias node.
    """
    return torch.stack([net_encode_fn(x) for x in X])
