import math

import torch
import torch.nn as nn


class PositionEncoding2D(nn.Module):
    def __init__(
        self,
        num_pos_feats=64,
        temperature=10000,
        normalize=False,
        scale=None,
        maxh=16,
        maxw=16,
    ):
        """This is a more standard version of the position embedding, very
        similar to the one used by the Attention is all you need paper,
        generalized to work on images.

        demision of output: (batch_size, num_pos_feats * 2, maxH, maxW)
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.maxh = maxh
        self.maxw = maxw
        pe = self._gen_pos_buffer()
        self.register_buffer("pe", pe)

    def _gen_pos_buffer(self):
        _eyes = torch.ones((1, self.maxh, self.maxw))
        y_embed = _eyes.cumsum(1, dtype=torch.float32)
        x_embed = _eyes.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, inp):
        """Generate positional encoding without added to original embedding."""
        x = inp
        return self.pe.repeat((x.size(0), 1, 1, 1))
