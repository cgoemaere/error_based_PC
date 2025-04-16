from functools import partial

import torchvision
import torchvision.transforms.v2 as v2

from .torchvision_datamodule import TorchvisionDataModule

__all__ = [
    "EMNIST",
    "FashionMNIST",
    "CIFAR10",
    "CIFAR100",
]


class EMNIST(TorchvisionDataModule):
    known_shapes = {"img": (1, 28, 28), "y": (10,)}
    transforms = [v2.Normalize(mean=[0.5], std=[0.5])]
    dataset = torchvision.datasets.EMNIST

    def __init__(self, batch_size: int = 64, split: str = "mnist"):
        super().__init__(batch_size)
        self.split = split

    def setup(self, stage: str):
        # Only now can we finally select the proper EMNIST split
        self.dataset = partial(self.dataset, split=self.split)
        super().setup(stage)


class FashionMNIST(TorchvisionDataModule):
    known_shapes = {"img": (1, 28, 28), "y": (10,)}
    transforms = [v2.Normalize(mean=[0.5], std=[0.5])]
    dataset = torchvision.datasets.FashionMNIST


class CIFAR10(TorchvisionDataModule):
    known_shapes = {"img": (3, 32, 32), "y": (10,)}
    transforms = [
        v2.RandomHorizontalFlip(p=0.5),
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ]
    dataset = torchvision.datasets.CIFAR10
    dl_kwargs = {"num_workers": 4, "pin_memory": True, "persistent_workers": True}


class CIFAR100(TorchvisionDataModule):
    known_shapes = {"img": (3, 32, 32), "y": (100,)}
    transforms = [
        v2.RandomHorizontalFlip(p=0.5),
        v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ]
    dataset = torchvision.datasets.CIFAR100
    dl_kwargs = {"num_workers": 4, "pin_memory": True, "persistent_workers": True}
