import torch
import torch.nn as nn
import torch.nn.functional as F
Tensor = torch.Tensor


class ResBlock(nn.Module):
    expansion = 1
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
    ) -> None:
        super().__init__()

        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if downsample or in_channels != out_channels:
            self.downsample = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
    
    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out += identity
        out = F.relu(out)
        return x


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
    ) -> None:
        super().__init__()

        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=(out_channels * self.expansion),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        if downsample:
            self.downsample = nn.Conv2d(
                in_channels=in_channels,
                out_channels=(out_channels * self.expansion),
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
    
    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv2(x)
        out = self.bn2(x)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out += identity
        out = F.relu(out)
        return out
