"""Train the FP32 TinyConvNet on FashionMNIST and save the weights.

Usage:
    python3 train.py --epochs 8

Produces:
    artifacts/model_fp32.pth
"""

import argparse
import os
import time

import torch
import torch.nn as nn

from data import get_loaders
from model import TinyConvNet


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out", type=str, default="artifacts/model_fp32.pth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")  # keep on-device / edge story CPU-only

    train_loader, test_loader = get_loaders(batch_size=args.batch_size)

    model = TinyConvNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        scheduler.step()
        train_loss = running / len(train_loader.dataset)
        acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:2d}/{args.epochs} | loss {train_loss:.4f} "
            f"| test acc {acc:.2f}% | {time.time() - start:.1f}s"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    final_acc = evaluate(model, test_loader, device)
    print(f"Saved FP32 model to {args.out} | final test acc {final_acc:.2f}%")


if __name__ == "__main__":
    main()
