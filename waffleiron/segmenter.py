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

import torch.nn as nn

from .backbone import WaffleIron
from .embedding import Embedding


class Segmenter(nn.Module):
    def __init__(
        self,
        input_channels,
        feat_channels,
        nb_class,
        depth,
        grid_shape,
        drop_path_prob=0,
        gelu=False,
        mlp_classif=False,
        mlp_hidden_size=None,
        checkpointing=False,
    ):
        super().__init__()

        # Embedding layer
        self.embed = Embedding(input_channels, feat_channels)

        # WaffleIron backbone
        self.waffleiron = WaffleIron(
            feat_channels,
            depth,
            grid_shape,
            drop_path_prob,
            gelu,
            checkpointing,
        )

        # Classification layer
        if mlp_classif:
            self.classif = nn.Sequential(
                nn.Conv1d(feat_channels, mlp_hidden_size, 1),
                nn.ReLU(inplace=True),
                nn.Conv1d(mlp_hidden_size, nb_class, 1),
            )
        else:
            self.classif = nn.Conv1d(feat_channels, nb_class, 1)

    def forward(self, feats, cell_ind, occupied_cell, neighbors):
        tokens = self.embed(feats, neighbors)
        tokens = self.waffleiron(tokens, cell_ind, occupied_cell)
        return self.classif(tokens)
