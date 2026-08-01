"""Compact, quantization-friendly CNN for FashionMNIST (1x28x28, 10 classes).

The architecture is intentionally small so it trains on a CPU in a few minutes,
and it is built to be quantization-friendly:
  * QuantStub / DeQuantStub bracket the network so static INT8 post-training
    quantization can insert observers at the tensor boundaries.
  * Each conv is followed by BatchNorm + ReLU so the (conv, bn, relu) triples can
    be fused before quantization, which both speeds up inference and improves the
    accuracy retained after INT8 conversion.
"""

import torch
import torch.nn as nn


class TinyConvNet(nn.Module):
    """~99K param CNN: 3 conv blocks + 2 FC layers.

    Layout:
      Block1: Conv(1->16)  -> BN -> ReLU -> MaxPool  (28 -> 14)
      Block2: Conv(16->32) -> BN -> ReLU -> MaxPool  (14 -> 7)
      Block3: Conv(32->64) -> BN -> ReLU -> MaxPool  (7  -> 3)
      Head:   FC(64*3*3 -> 128) -> ReLU -> FC(128 -> 10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.quant(x)

        x = self.pool(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool(self.relu3(self.bn3(self.conv3(x))))

        x = torch.flatten(x, 1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)

        x = self.dequant(x)
        return x

    def fuse_modules(self):
        """Fuse (conv, bn, relu) and (linear, relu) for quantization."""
        torch.quantization.fuse_modules(
            self,
            [
                ["conv1", "bn1", "relu1"],
                ["conv2", "bn2", "relu2"],
                ["conv3", "bn3", "relu3"],
                ["fc1", "relu4"],
            ],
            inplace=True,
        )


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
