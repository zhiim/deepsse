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


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.functional.relu
    if activation == "gelu":
        return nn.functional.gelu
    if activation == "glu":
        return nn.functional.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class CALayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        """Cross attention layer, attention + FFN like Transformer's decoder."""
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        query,
        feature,
        pos,
    ):
        # -- cross-attention ---------------------------------------------------
        tgt1, _ = self.multihead_attn(
            query=query,
            key=feature + pos,
            value=feature,
        )

        # -- FFN ---------------------------------------------------------------
        tgt = query + self.dropout1(tgt1)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        return tgt


class CABlock(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        query,
        feature,
        pos: Optional[Tensor] = None,
    ):
        output = query

        for layer in self.layers:
            output = layer(
                output,
                feature,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class AngularGridSearch(BaseModule):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_ca_layers=1,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()

        ca_layer = CALayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

        norm = nn.LayerNorm(d_model)
        self.ca_block = CABlock(ca_layer, num_ca_layers, norm)

        self.d_model = d_model

    def forward(self, feature, query, pos_embed):
        # feature: (HW, bs, d)
        # query: (num_class, bs, d)
        # pos_embed: (HW, bs, d)

        out = self.ca_block(
            query,  # (num_clas, bs, d)
            feature,  # feature from SFE, (hw, bs, d)
            pos=pos_embed,  # pos encoding of feature
        )

        # hs: (bs, K, d)
        return out.transpose(0, 1)


class FeatureWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))

    def forward(self, x):
        # x: (bs, num_class, d)
        # self.W: (1, num_class, d)
        x = (self.W * x).sum(-1)
        # point-wise mul and sum in the last dim, x will be (bs, num_class)

        if self.bias:
            x = x + self.b
            # broadcast when adding bias, x will be (bs, num_class)

        return x


class TransForm(BaseModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.input_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.flatten(2).permute(2, 0, 1)

        return x


class DeepSSE(BaseModel):
    def __init__(self, num_class, num_antenna, antenna_spacing, **kwargs):
        """Deep Learning based Spatial Spectrum Estimator."""
        super().__init__()
        self.sfe = SpatialFeatureExtractor.build_model(**kwargs)

        self.ags = AngularGridSearch.build_model(**kwargs)

        self.pos_embed = PositionEncoding2D(
            num_pos_feats=self.ags.d_model // 2,
            maxh=num_antenna,
            maxw=num_antenna,
        )

        hidden_dim = self.ags.d_model

        self.transform = TransForm(self.sfe.out_channels, hidden_dim)

        self.angle_projector = AngleFeatureProjector(
            3 * num_antenna, hidden_dim
        )

        self.fc = FeatureWiseLinear(num_class, hidden_dim, bias=True)

        self._get_steering_vectors(num_class, num_antenna, antenna_spacing)

    def _get_steering_vectors(self, num_class, num_antenna, antenna_spacing):
        grids = torch.linspace(-90, 90 - 180 / num_class, num_class)
        antenna_position = (
            (torch.arange(0, num_antenna, 1) * antenna_spacing)
            .view(-1, 1)
            .to(torch.float)
        )
        delay = antenna_position @ torch.sin(grids).view(1, -1)

        steering_vetor = torch.exp(-2j * math.pi * delay)
        steering_vetor = torch.cat(
            (steering_vetor.real, steering_vetor.imag, steering_vetor.angle()),
            dim=0,
        )

        self.register_buffer("angle_embed", steering_vetor.transpose(0, 1))

    def forward(self, x):
        # inp: (bs, C, H_0, W_0)
        bs, _, _, _ = x.shape

        spatial_feature = self.sfe(x)  # (bs, d_0, H, W)

        angle_embed = self.angle_projector(self.angle_embed)  # (num_class, d)
        angle_embed = angle_embed.unsqueeze(1).repeat(
            1, bs, 1
        )  # (num_class, bs, d)

        pos_embed = self.pos_embed(
            spatial_feature
        )  # pos_embed of spatial feature: (bs, 2*d//2, H, W)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # (HW, bs, d)

        out = self.ags(
            feature=self.transform(spatial_feature),
            query=angle_embed,
            pos_embed=pos_embed,
        )

        out = self.fc(out)

        return torch.sigmoid(out)
