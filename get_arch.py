from torch import nn


def get_architecture(dataset: str):

    if dataset == "EMNIST":
        architecture = [
            nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 512), nn.Tanh()),
            nn.Sequential(nn.Linear(512, 512), nn.Tanh()),
            nn.Sequential(nn.Linear(512, 10), nn.Sigmoid()),
        ]
    elif dataset == "CIFAR10":
        architecture = [
            nn.Sequential(nn.Conv2d(3, 128, 3, 1, 1), nn.MaxPool2d(2, 2), nn.GELU()),
            nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.MaxPool2d(2, 2), nn.GELU()),
            nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.MaxPool2d(2, 2), nn.GELU()),
            nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.MaxPool2d(2, 2), nn.GELU()),
            nn.Sequential(nn.Flatten(), nn.Linear(2048, 128, bias=True), nn.GELU()),
            nn.Sequential(nn.Linear(128, 10, bias=True), nn.Sigmoid()),
        ]

    return architecture
