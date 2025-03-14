import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import List, Callable

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(
        self,
        in_planes: int,
        planes: int,
        activation: Callable,
        stride: int=1,
    ) -> None:
        super(BasicBlock, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion*planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        activation: Callable,
        stride: int=1,
    ) -> None:
        super(Bottleneck, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        num_blocks: List[int],
        in_dim: int,
        out_dim: int,
        activation: Callable, 
    ) -> None:
        super(ResNet, self).__init__()
        self.activation = activation
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, out_dim)

    def _make_layer(
        self,
        block: nn.Module,
        planes: int,
        num_blocks: int,
        stride: int
    ) -> nn.Module:
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.activation, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(in_dim: int, out_dim: int, activation: Callable) -> nn.Module:
    return ResNet(BasicBlock, [2, 2, 2, 2], in_dim, out_dim, activation)


def resnet34(in_dim: int, out_dim: int, activation: Callable) -> nn.Module:
    return ResNet(BasicBlock, [3, 4, 6, 3], in_dim, out_dim, activation)


def resnet50(in_dim: int, out_dim: int, activation: Callable) -> nn.Module:
    return ResNet(Bottleneck, [3, 4, 6, 3], in_dim, out_dim, activation)


def resnet101(in_dim: int, out_dim: int, activation: Callable) -> nn.Module:
    return ResNet(Bottleneck, [3, 4, 23, 3], in_dim, out_dim, activation)
