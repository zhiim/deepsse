# +----------------------------------------------------------+
# | we will add other parts as soon as we finish the review  |
# +----------------------------------------------------------+

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from model.base_model import BaseModel, BaseModule
from model.custom_module.layers import ResidualBlock, _get_activation
from model.custom_module.module import PositionEncoding2D


class SpatialFeatureExtractor(BaseModule):
    def __init__(
        self,
        img_channels,
        layers=[2, 2, 2, 2],
        in_channels=32,
        out_channels=32,
        backbone_activation="relu",
    ):
        """ResNet like architecture for spatial feature extraction."""
        super().__init__()

        # layers = [2, 2, 2, 2] is resnet18
        self.expansion = 1

        self.activation = backbone_activation

        self.in_channels = in_channels
        self.out_channels = out_channels

        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.input_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=img_channels,
                out_channels=self.in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(self.in_channels),
            _get_activation(self.activation),
        )

        self.res_layers = nn.ModuleList()
        for num_layer in layers:
            self.res_layers.append(
                self._make_layer(self.out_channels, num_layer)
            )

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(
            ResidualBlock(
                self.in_channels,
                out_channels,
                stride=stride,
                activation=self.activation,
            )
        )

        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(
                ResidualBlock(
                    self.in_channels, out_channels, activation=self.activation
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layers(x)

        for res_layer in self.res_layers:
            x = res_layer(x)

        return x


class AngleFeatureProjector(BaseModule):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.angle_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return self.angle_proj(x)
