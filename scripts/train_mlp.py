"""Train a small MLP on MNIST and save weights for downstream sweep experiments.

Network: Linear(784,256) -> ReLU -> Linear(256,128) -> ReLU -> Linear(128,10)
No Flatten layer — inputs are pre-flattened so the model is directly compatible
with DecomposedMLP.from_pytorch_sequential.

Usage:
    uv run python scripts/train_mlp.py
    uv run python scripts/train_mlp.py --epochs 5 --output data/mlp_mnist.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_mlp() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def evaluate(model: nn.Sequential, loader: DataLoader) -> float:  # type: ignore[type-arg]
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.reshape(x.size(0), -1)
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


def train(
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    data_dir: Path = Path("data"),
    output: Path = Path("data/mlp_mnist.pt"),
    seed: int = 42,
) -> nn.Sequential:
    torch.manual_seed(seed)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(str(data_dir), train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = build_mlp()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.reshape(x.size(0), -1)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        acc = evaluate(model, test_loader)
        avg_loss = running_loss / len(train_loader)
        print(f"epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  test_acc={acc:.4f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    print(f"\nSaved weights → {output}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP on MNIST")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/mlp_mnist.pt"))
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
