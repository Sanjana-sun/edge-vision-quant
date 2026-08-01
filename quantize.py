"""Post-training static INT8 quantization of the trained FP32 model.

Static PTQ pipeline:
    1. Load FP32 weights.
    2. Fuse (conv, bn, relu) / (linear, relu) modules.
    3. Attach a qconfig (qnnpack, the ARM / mobile backend).
    4. Insert observers (prepare) and run a calibration pass over a subset of
       the training data so activation ranges are recorded.
    5. Convert to a true INT8 model.

Usage:
    python3 quantize.py

Produces:
    artifacts/model_int8.pth   (quantized state dict)
"""

import argparse
import os

import torch

from data import get_loaders
from model import TinyConvNet


def build_quantized_model(fp32_path: str, calib_batches: int = 20):
    torch.backends.quantized.engine = "qnnpack"

    model = TinyConvNet()
    model.load_state_dict(torch.load(fp32_path, map_location="cpu"))
    model.eval()

    # Fuse conv-bn-relu etc. for accurate, fast INT8 inference.
    model.fuse_modules()

    # qnnpack == ARM / mobile INT8 backend (the on-device target).
    model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
    torch.quantization.prepare(model, inplace=True)

    # Calibration: feed real data so activation observers learn value ranges.
    train_loader, _ = get_loaders(batch_size=128)
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            model(x)
            if i + 1 >= calib_batches:
                break

    torch.quantization.convert(model, inplace=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", type=str, default="artifacts/model_fp32.pth")
    parser.add_argument("--out", type=str, default="artifacts/model_int8.pth")
    parser.add_argument("--calib-batches", type=int, default=20)
    args = parser.parse_args()

    model = build_quantized_model(args.fp32, args.calib_batches)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved INT8 quantized model to {args.out}")


if __name__ == "__main__":
    main()
