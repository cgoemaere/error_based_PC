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
        v2.RandomCrop(32, padding=4),
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ]
    dataset = torchvision.datasets.CIFAR10
    # dl_kwargs = {"num_workers": 0, "pin_memory": False, "persistent_workers": False}
    dl_kwargs = {"num_workers": 8, "pin_memory": True, "persistent_workers": True}


class CIFAR100(TorchvisionDataModule):
    known_shapes = {"img": (3, 32, 32), "y": (100,)}
    transforms = [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomCrop(32, padding=4),
        v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ]
    dataset = torchvision.datasets.CIFAR100
    dl_kwargs = {"num_workers": 0, "pin_memory": False, "persistent_workers": False}
    # dl_kwargs = {"num_workers": 7, "pin_memory": True, "persistent_workers": True}


import os
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class TinyImageNetValDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.loader = default_loader

        # Load annotations
        val_annotations_file = os.path.join(root, 'val_annotations.txt')
        with open(val_annotations_file, 'r') as f:
            lines = f.readlines()

        self.imgs = []
        for line in lines:
            parts = line.strip().split('\t')
            filename, label = parts[0], parts[1]
            img_path = os.path.join(root, 'images', filename)
            self.imgs.append((img_path, label))

        # Create class-to-index map
        classes = sorted(set(label for _, label in self.imgs))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.targets = [self.class_to_idx[label] for _, label in self.imgs]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, label = self.imgs[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        target = self.class_to_idx[label]
        return image, target

from torch.utils.data import DataLoader, TensorDataset
class TinyImageNet(TorchvisionDataModule):
    known_shapes = {"img": (3, 64, 64), "y": (200,)}
    transforms = [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomCrop(64, padding=4),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    # dl_kwargs = {"num_workers": 0, "pin_memory": False, "persistent_workers": False}
    dl_kwargs = {"num_workers": 8, "pin_memory": True, "persistent_workers": True}

    def __init__(self, batch_size: int = 64, data_dir: str = "../data/tiny-imagenet-200"):
        super().__init__(batch_size)
        self.data_dir = data_dir

    def setup(self, stage: str):
        self.num_classes = 200
        transform = v2.Compose([v2.ToTensor(), *self.transforms])
        if stage == "fit" or stage is None:
            ## standard way to load the dataset, takes no setup for 20min/epoch
            # self.train_set = torchvision.datasets.ImageFolder(root = f"{self.data_dir}/train", transform=transform)
            # self.val_set = TinyImageNetValDataset(root=f"{self.data_dir}/val", transform=transform)

            # loads the entire dataset into cpu memory - takes about 128GB, takes 20 minutes at setup for 6/7min/epoch
            train_set = torchvision.datasets.ImageFolder(root = f"{self.data_dir}/train", transform=transform)
            val_set = TinyImageNetValDataset(root=f"{self.data_dir}/val", transform=transform)
            train_cpu_dl = DataLoader(train_set, batch_size=train_set.__len__())
            val_cpu_dl = DataLoader(val_set, batch_size=val_set.__len__())

            # make dataset with in memory data
            train_cpu = next(iter(train_cpu_dl))
            val_cpu = next(iter(val_cpu_dl))

            self.train_set = TensorDataset(train_cpu[0], train_cpu[1])
            self.val_set = TensorDataset(val_cpu[0], val_cpu[1])
        elif stage == "test":
            self.test_set = self.val_set

    @property
    def dataset_name(self):
        return "tiny-imagenet"
