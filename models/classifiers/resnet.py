import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNet18MNIST(nn.Module):
    def __init__(self, num_classes: int = 10, channels: int = 32):
        super().__init__()

        self.in_channels = channels

        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(channels, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(channels * 2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(channels * 4, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(channels * 8, num_blocks=2, stride=2)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels * 8 * BasicBlock.expansion, num_classes),
        )

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels * BasicBlock.expansion

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(x)


def build_victim(config) -> ResNet18MNIST:
    arch = str(getattr(config, "victim_arch", "resnet18_mnist")).lower()
    if arch != "resnet18_mnist":
        raise NotImplementedError("Only victim_arch=resnet18_mnist is currently supported.")
    return ResNet18MNIST(
        num_classes=int(getattr(config, "num_classes", 10)),
        channels=int(getattr(config, "victim_channels", 32)),
    )
