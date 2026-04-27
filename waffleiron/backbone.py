# Copyright 2026 - Valeo Comfort and Driving Assistance - valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .helper_projection import get_all_projections_scatter_reduce as get_all_projections
from .helper_projection import projection_3d_to_2d_scatter_reduce as projection_3d_to_2d


class myLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, **kwargs):
        return super().forward(x.transpose(1, -1)).transpose(1, -1)


class DropPath(nn.Module):
    """
    Stochastic Depth

    Original code of this module is at:
    https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob

    def extra_repr(self):
        return f"prob={self.drop_prob}"

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()  # binarize
        output = x.div(self.keep_prob) * random_tensor
        return output


class ChannelMix(nn.Module):
    def __init__(
        self,
        channels,
        drop_path_prob,
        gelu=False,
    ):
        super().__init__()

        self.norm = myLayerNorm(channels)

        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.GELU() if gelu else nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 1),
        )

        self.scale = nn.Conv1d(
            channels,
            channels,
            1,
            bias=False,
            groups=channels,
        )  # Implement LayerScale

        self.drop_path = DropPath(drop_path_prob)

    def forward(self, tokens):
        """tokens <- tokens + LayerScale( MLP( BN(tokens) ) )"""
        residual = self.norm(tokens)
        residual = self.mlp(residual)
        return tokens + self.drop_path(self.scale(residual))


class SpatialMix(nn.Module):
    def __init__(
        self,
        channels,
        grid_shape,
        drop_path_prob,
        gelu=False,
    ):
        super().__init__()
        self.H, self.W = grid_shape

        self.norm = myLayerNorm(channels)

        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU() if gelu else nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
        )

        self.scale = nn.Conv1d(
            channels, channels, 1, bias=False, groups=channels
        )  # Implement LayerScale

        self.drop_path = DropPath(drop_path_prob)

    def extra_repr(self):
        return f"(grid): [{self.H}, {self.W}]"

    def forward(self, tokens, sp_mat):
        """tokens <- tokens + LayerScale( Inflate( FFN( Flatten( BN(tokens) ) ) )"""
        B, C, N = tokens.shape

        # Norm
        residual = self.norm(tokens)

        # Flatten
        residual = projection_3d_to_2d(residual, sp_mat, B, C, self.H, self.W)
        residual = residual.reshape(B, C, self.H, self.W)

        # FFN
        residual = self.ffn(residual)

        # LayerScale
        residual = residual.reshape(B, C, self.H * self.W)
        residual = self.scale(residual)

        # Inflate
        residual = torch.gather(residual, 2, sp_mat["inflate"])

        return tokens + self.drop_path(residual)


class WaffleIron(nn.Module):
    def __init__(
        self,
        channels,
        depth,
        grids_shape,
        drop_path_prob,
        gelu=False,
        checkpointing=False,
    ):
        super().__init__()
        self.depth = depth
        self.grids_shape = grids_shape
        self.checkpointing = checkpointing

        self.channel_mix = nn.ModuleList(
            [
                ChannelMix(
                    channels,
                    drop_path_prob,
                    gelu,
                )
                for _ in range(depth)
            ]
        )

        self.spatial_mix = nn.ModuleList(
            [
                SpatialMix(
                    channels,
                    grids_shape[d % len(grids_shape)],
                    drop_path_prob,
                    gelu,
                )
                for d in range(depth)
            ]
        )

    def forward(self, tokens, cell_ind, occupied_cell):
        # Build all 3D to 2D projection matrices
        batch_size, nb_feat, num_points = tokens.shape
        sp_mat = get_all_projections(
            cell_ind,
            nb_feat,
            batch_size,
            num_points,
            occupied_cell,
            tokens.device,
            self.grids_shape,
        )

        # Actual backbone
        for d, (smix, cmix) in enumerate(zip(self.spatial_mix, self.channel_mix)):
            if self.checkpointing:
                tokens = checkpoint(
                    smix, tokens, sp_mat[d % len(sp_mat)], use_reentrant=False
                )
                tokens = checkpoint(cmix, tokens, use_reentrant=False)
            else:
                tokens = smix(tokens, sp_mat[d % len(sp_mat)])
                tokens = cmix(tokens)

        return tokens
