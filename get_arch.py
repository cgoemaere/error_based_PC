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


def get_cnn_architecture(model_name, dataset, activation):
    if dataset == "CIFAR10":
        num_classes = 10
        img_size = 32
    elif dataset == "CIFAR100":
        num_classes = 100
        img_size = 32
    elif dataset == "tiny-imagenet":
        num_classes = 200
        img_size = 64
    else:
        raise ValueError("Unsupported dataset. Only CIFAR10, cifar100 and tiny-imagenet are supported.")

    if activation == "relu":
        activation = nn.ReLU
    elif activation == "gelu":
        activation = nn.GELU
    elif activation == "tanh":
        activation = nn.Tanh
    elif activation == "leaky_relu":
        activation = nn.LeakyReLU
    else:
        raise ValueError("Unsupported activation function. Only relu, gelu, tanh and leaky_relu are supported.")


    if model_name == "VGG5":
        architecture = [
            nn.Sequential(nn.Conv2d(3, 128, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Flatten(), nn.Linear(512 * (img_size // 2**4)**2 , num_classes, bias=True)),
        ]
    elif model_name == "VGG7":
        architecture = [
            nn.Sequential(nn.Conv2d(3, 128, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), activation()),
            nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(256, 256, 3, 1, 0), activation()),
            nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(512, 512, 3, 1, 0), activation()),
            nn.Sequential(nn.Flatten(), nn.Linear(512 * (img_size // 2**5)**2, num_classes, bias=True)),
        ]
    elif model_name == "VGG9": 
        architecture = [
            nn.Sequential(nn.Conv2d(3, 128, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), activation()),
            nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), activation()),
            nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), activation()),
            nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), activation()),
            nn.Sequential(nn.Flatten(), nn.Linear(512 * (img_size // 2**4)**2, num_classes, bias=True)),
        ]
    elif model_name == "VGG11":
        architecture = [
            nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), activation()),
            nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), activation()),
            nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), activation()),
            nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Flatten(), nn.Linear(512 * (img_size // 2**5)**2, num_classes)),
        ]

    elif model_name == "VGG13":
        architecture = [
            nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), activation()),
            nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), activation()),
            nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), activation()),
            nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), activation()),
            nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), activation()),
            nn.Sequential(nn.Flatten(), nn.Linear(512 * (img_size // 2**5)**2, 512 * (img_size // 2**5)**2), activation()),
            nn.Sequential(nn.Linear(512 * (img_size // 2**5)**2, 1000), activation()),
            nn.Sequential(nn.Linear(1000, num_classes)),
        ]

    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return architecture


class SaveIdentity(nn.Module):
    def __init__(self, identity_downsample=None):
        super(SaveIdentity, self).__init__()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        if isinstance(x, tuple):
            if self.identity_downsample is not None:
                x_new = self.identity_downsample(x[0])
            else:
                x_new = x[0]
            x = (x[0], x_new)
        return x

class AddIdentity(nn.Module):
    def __init__(self):
        super(AddIdentity, self).__init__()
        
    def forward(self, x):
        if isinstance(x, tuple):
            x = (x[0] + x[1], 0.)
        return x

class LayerWithResidual(nn.Module):
    def __init__(self, layer):
        super(LayerWithResidual, self).__init__()
        self.layer = layer
        
    def forward(self, x):
        if isinstance(x, tuple):
            y = self.layer(x[0])
            return (y, x[1])
        else:
            return self.layer(x)

def get_resnet_block( in_channels, out_channels, stride=1, avg_pool=False):
    """
    Returns a ResNet block with the specified parameters.
    """
    conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    bn1 = nn.BatchNorm2d(out_channels)
    conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    bn2 = nn.BatchNorm2d(out_channels)
    relu = nn.ReLU()

    if stride > 1:
        identity_downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )
    else:
        identity_downsample = None        

    block1 = nn.Sequential(
        SaveIdentity(identity_downsample),
        LayerWithResidual(conv1),
        LayerWithResidual(bn1),
        LayerWithResidual(relu),
    )

    block2 = nn.Sequential(
        LayerWithResidual(conv2),
        LayerWithResidual(bn2),
        AddIdentity(),
        LayerWithResidual(relu),
        *([LayerWithResidual(nn.AvgPool2d(kernel_size=4, stride=4))] if avg_pool else []),
    )

    return block1, block2




def get_resnet_architecture(model_name, dataset):
    if dataset == "CIFAR10":
        num_classes = 10
        img_size = 32
    elif dataset == "CIFAR100":
        num_classes = 100
        img_size = 32
    elif dataset == "tiny-imagenet":
        num_classes = 200
        img_size = 64
    else:
        raise ValueError("Unsupported dataset. Only CIFAR10, cifar100 and tiny-imagenet are supported.")

    activation = nn.ReLU

    if model_name == "ResNet18":
        architecture = [
            LayerWithResidual(nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), activation())),
            *get_resnet_block(64, 64, stride=1),
            *get_resnet_block(64, 128, stride=2),
            *get_resnet_block(128, 256, stride=2),
            *get_resnet_block(256, 512, stride=2, avg_pool=True),
            LayerWithResidual(nn.Sequential(nn.Flatten(), nn.Linear(512 * (img_size // 2**5)**2 , num_classes))),
        ]
    
    return architecture
