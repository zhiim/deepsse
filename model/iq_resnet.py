import torch
import torch.nn as nn

from model.base_model import BaseModel


class ResLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        """Residual Block"""
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 3),
            stride=(1, 2),
            padding=(0, 1),
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 2),
            padding=(0, 0),
        )

    def forward(self, x):
        identity = self.conv3(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class IQResNet(BaseModel):
    def __init__(self, num_classes=180, num_antennas=16):
        """
        Reference:
            Zheng, Shilian, Zhuang Yang, Weiguo Shen, Luxin Zhang, Jiawei Zhu,
            Zhijin Zhao, and Xiaoniu Yang. “Deep Learning-Based DOA Estimation.”
            IEEE Transactions on Cognitive Communications and Networking, 2024,
            1–1. https://doi.org/10.1109/TCCN.2024.3360527.
        """
        super().__init__()

        # input (32, 300)

        self.f1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(2 * num_antennas, 5),
                stride=(2 * num_antennas, 1),
                padding=(0, 0),
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )  # (1, 296)

        self.f2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))  # (1, 147)

        self.f3 = ResLayer(in_channels=64, out_channels=64)  # (1, 74)

        self.f4 = ResLayer(in_channels=64, out_channels=128)  # (1, 37)

        self.f5 = ResLayer(in_channels=128, out_channels=256)  # (1, 19)

        self.f6 = ResLayer(in_channels=256, out_channels=512)  # (1, 10)

        self.fc = nn.Linear(512, num_classes)

        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)

        x = torch.mean(x, dim=-1).squeeze(-1)
        x = self.fc(x)
        x = self.out(x)
        return x
