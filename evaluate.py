"""Benchmark FP32 vs INT8: test accuracy, on-disk size, single-sample latency.

Usage:
    python3 evaluate.py

Prints a results table and writes results.md.
"""

import argparse
import os
import time

import torch

from data import get_loaders
from model import TinyConvNet


def load_fp32(path):
    model = TinyConvNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def load_int8(path):
    """Rebuild the quantized skeleton, then load the INT8 weights into it."""
    torch.backends.quantized.engine = "qnnpack"
    model = TinyConvNet()
    model.eval()
    model.fuse_modules()
    model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def accuracy(model, loader):
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def latency_ms(model, sample, warmup=20, runs=200):
    """Mean single-sample (batch size 1) CPU inference latency in ms."""
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        start = time.perf_counter()
        for _ in range(runs):
            model(sample)
        elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", type=str, default="artifacts/model_fp32.pth")
    parser.add_argument("--int8", type=str, default="artifacts/model_int8.pth")
    args = parser.parse_args()

    torch.manual_seed(0)
    _, test_loader = get_loaders(batch_size=256)

    fp32 = load_fp32(args.fp32)
    int8 = load_int8(args.int8)

    sample = next(iter(test_loader))[0][:1]  # single 1x1x28x28 image

    fp32_acc = accuracy(fp32, test_loader)
    int8_acc = accuracy(int8, test_loader)

    fp32_size = file_size_mb(args.fp32)
    int8_size = file_size_mb(args.int8)

    fp32_lat = latency_ms(fp32, sample)
    int8_lat = latency_ms(int8, sample)

    size_x = fp32_size / int8_size
    lat_x = fp32_lat / int8_lat
    acc_delta = int8_acc - fp32_acc  # negative => accuracy dropped after INT8

    rows = [
        ("Metric", "FP32", "INT8", "Change"),
        ("Test accuracy (%)", f"{fp32_acc:.2f}", f"{int8_acc:.2f}", f"{acc_delta:+.2f} pts"),
        ("Model size (MB)", f"{fp32_size:.3f}", f"{int8_size:.3f}", f"{size_x:.2f}x smaller"),
        ("Latency (ms/img)", f"{fp32_lat:.3f}", f"{int8_lat:.3f}", f"{lat_x:.2f}x faster"),
    ]

    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    print()
    for idx, r in enumerate(rows):
        line = " | ".join(r[i].ljust(widths[i]) for i in range(4))
        print(line)
        if idx == 0:
            print("-+-".join("-" * widths[i] for i in range(4)))
    print()

    with open("results.md", "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write("FashionMNIST test set (10,000 images). CPU, qnnpack INT8 backend.\n\n")
        f.write("| Metric | FP32 | INT8 | Change |\n")
        f.write("|---|---|---|---|\n")
        for r in rows[1:]:
            f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |\n")
        f.write(
            f"\nINT8 holds accuracy at **{int8_acc:.2f}%** "
            f"({acc_delta:+.2f} pts vs FP32) while being "
            f"**{size_x:.2f}x smaller** and **{lat_x:.2f}x faster** per image.\n"
        )
    print("Wrote results.md")


if __name__ == "__main__":
    main()
