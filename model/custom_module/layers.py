import torch.nn as nn
from torch import Tensor


def _get_activation(activation: str = "relu"):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "prelu":
        return nn.PReLU()
    raise ValueError(f"Activation {activation} not supported")


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module | None = None,
        activation: str = "relu",
    ):
        """Residual Block

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride of the first convolutional layer
            expansion: multiplicative factor for the subsequent conv2d layer's
                output
            downsample: downsample layer
        """
        super().__init__()

        # Multiplicative factor for the subsequent conv2d layer's output
        # channels. It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        # 第一层卷积
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = _get_activation(activation)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
