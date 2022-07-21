import torch
from torch import Tensor
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != self.expansion * channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.bn1(self.conv1(x))
        out = self.relu(out)

        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, channels: int, stride: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, self.expansion * channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != self.expansion * channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.bn1(self.conv1(x))
        out = self.relu(out)

        out = self.bn2(self.conv2(out))
        out = self.relu(out)

        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num: int, classes: int = 1000):
        super().__init__()

        self.num = num
        self.block = {
            18: BasicBlock,
            34: BasicBlock,
            50: Bottleneck,
            101: Bottleneck,
            152: Bottleneck,
        }[num]

        self.num_blocks = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[num]

        self.in_channels = 64

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block.expansion, classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x

    def load_from_pretrained(self):
        model_urls = {
            18: "https://download.pytorch.org/models/resnet18-f37072fd.pth",
            34: "https://download.pytorch.org/models/resnet34-b627a593.pth",
            50: "https://download.pytorch.org/models/resnet50-0676ba61.pth",
            101: "https://download.pytorch.org/models/resnet101-63fe2227.pth",
            152: "https://download.pytorch.org/models/resnet152-394f9c45.pth",
        }
        url = model_urls[self.num]
        state_dict = torch.hub.load_state_dict_from_url(url)
        self.load_state_dict(state_dict)

    def _make_layer(self, block, channels: int, num_blocks: int, stride: int):
        # Only the first block is done with the specified stride
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = block.expansion * channels
        return nn.Sequential(*layers)


ResNet18 = lambda classes=1000: ResNet(18, classes)
ResNet34 = lambda classes=1000: ResNet(34, classes)
ResNet50 = lambda classes=1000: ResNet(50, classes)
ResNet101 = lambda classes=1000: ResNet(101, classes)
ResNet152 = lambda classes=1000: ResNet(152, classes)