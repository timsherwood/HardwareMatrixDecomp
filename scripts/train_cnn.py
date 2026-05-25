"""Train a small CNN on Fashion-MNIST and save weights for downstream sweep experiments.

Fashion-MNIST: 10 clothing categories (T-shirt, trouser, pullover, dress, coat,
sandal, shirt, sneaker, bag, ankle boot). Same 28×28 grayscale format as MNIST
but significantly harder (~90% vs ~99% accuracy), making it a more meaningful
benchmark for noise sensitivity analysis.

Network:
  Conv2d(1,16,3,p=1) → ReLU → MaxPool2d(2)   → (16, 14, 14)
  Conv2d(16,32,3,p=1) → ReLU → MaxPool2d(2)  → (32, 7, 7)
  Flatten
  Linear(32*7*7, 128) → ReLU → Linear(128, 10)

All conv weight matrices stay well below the 100×100 tile limit:
  Layer 1: W shape (1*3*3, 16)  = (9, 16)
  Layer 2: W shape (16*3*3, 32) = (144, 32)  — largest column dim is 32 ≤ 100

Usage:
    uv run python scripts/train_cnn.py
    uv run python scripts/train_cnn.py --epochs 15 --output data/cnn_fmnist.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_cnn() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def evaluate(model: nn.Sequential, loader: DataLoader) -> float:  # type: ignore[type-arg]
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


def train(
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    data_dir: Path = Path("data"),
    output: Path = Path("data/cnn_mnist.pt"),
    seed: int = 42,
) -> nn.Sequential:
    torch.manual_seed(seed)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    train_ds = datasets.FashionMNIST(str(data_dir), train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(str(data_dir), train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = build_cnn()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        acc = evaluate(model, test_loader)
        avg_loss = running_loss / len(train_loader)
        print(f"epoch {epoch + 1:2d}/{epochs}  loss={avg_loss:.4f}  test_acc={acc:.4f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    print(f"\nSaved weights → {output}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN on MNIST")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/cnn_fmnist.pt"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_dir=args.data_dir,
        output=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
