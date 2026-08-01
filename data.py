"""FashionMNIST data loaders shared across train / quantize / evaluate."""

import torch
from torchvision import datasets, transforms

# Dataset mean/std for FashionMNIST (single channel).
_MEAN = (0.2860,)
_STD = (0.3530,)

_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]
)


def get_loaders(data_dir: str = "./data", batch_size: int = 128, num_workers: int = 2):
    train_set = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=_transform
    )
    test_set = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader
