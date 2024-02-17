# Copyright 2024 - Valeo Comfort and Driving Assistance - valeo.ai
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
import numpy as np
import utils.transforms as tr
from .pc_dataset import PCDataset, zero_pad
from scipy.spatial import cKDTree as KDTree
from torchvision.transforms.functional import resize as vision_resize


class CollateDistillation:
    def __init__(self, num_points=None):
        self.num_points = num_points
        assert num_points is None or num_points > 0

    def __call__(self, list_data):

        # Extract all data
        list_of_data = (list(data) for data in zip(*list_data))
        (
            feat,
            images,
            pairing_points,
            pairing_images,
            cell_ind,
            neighbors_emb,
        ) = list_of_data

        # Zero-pad point clouds
        Nmax = np.max([f.shape[-1] for f in feat])
        if self.num_points is not None:
            assert Nmax <= self.num_points
        occupied_cells = []
        for i in range(len(feat)):
            feat[i], neighbors_emb[i], cell_ind[i], temp = zero_pad(
                feat[i],
                neighbors_emb[i],
                cell_ind[i],
                Nmax if self.num_points is None else self.num_points,
            )
            occupied_cells.append(temp)

        # Concatenate along batch dimension
        feat = torch.from_numpy(np.vstack(feat)).float()  # B x C x Nmax
        images = torch.cat(images, 0).float()
        neighbors_emb = torch.from_numpy(np.vstack(neighbors_emb)).long()  # B x Nmax
        cell_ind = torch.from_numpy(
            np.vstack(cell_ind)
        ).long()  # B x nb_2d_cells x Nmax
        occupied_cells = torch.from_numpy(np.vstack(occupied_cells)).float()  # B x Nmax
        pairing_points = [torch.tensor(p) for p in pairing_points]
        pairing_images = [torch.tensor(p) for p in pairing_images]

        # Prepare output variables
        out = {
            "feat": feat,
            "images": images,
            "neighbors_emb": neighbors_emb,
            "cell_ind": cell_ind,
            "occupied_cells": occupied_cells,
            "pairing_points": pairing_points,
            "pairing_images": pairing_images,
        }

        return out


class ImPcDataset(PCDataset):
    """
    Dataset matching a 3D points cloud and an image using projection.
    """

    def __init__(self, max_points, im_size, **kwargs):
        super().__init__(**kwargs)

        assert self.phase == "train"

        self.im_size = im_size

        self.limit_num_points = tr.LimitNumPoints(
            dims=(0, 1, 2),
            max_point=max_points,
            random=True,
        )

        self.pc_augmentations = tr.Compose(
            [
                tr.Rotation(inplace=True, dim=2),
                tr.RandomApply(tr.FlipXY(inplace=True), prob=2 / 3),
                tr.Scale(inplace=True, dims=(0, 1, 2), range=0.1),
            ]
        )

    def load_pc(self, index):
        raise NotImplementedError()

    def map_pc_to_image(self, pc, index, min_dist=1.0):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def resize_im(self, im, pairing_images):
        # Rescale pixel coordinates
        rescale = [1.0, self.im_size[0] / im.shape[-2], self.im_size[1] / im.shape[-1]]
        pairing_images = np.floor(np.multiply(pairing_images, rescale))
        pairing_images = pairing_images.astype(np.int64)
        # Rescale image
        im = vision_resize(im, self.im_size)
        return im, pairing_images

    def __getitem__(self, index):
        # Load original point cloud
        pc = self.load_pc(index)

        # Voxelization
        pc, _ = self.downsample(pc, None)

        # Project point cloud to image
        pc, images, pairing_images = self.map_pc_to_image(pc, index)
        images = torch.tensor(np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))
        assert len(pairing_images) > 0

        # Limit number of points and ...
        pc, _, idx = self.limit_num_points(pc, None, return_idx=True)
        # ... adapt (points, pixels) pairs
        pairing_images = pairing_images[idx]

        # Apply augmentations
        pc, _ = self.pc_augmentations(pc, None)
        images, pairing_images = self.resize_im(images, pairing_images)

        # Crop to fov and ...
        pc, _, where = self.crop_to_fov(pc, None, return_mask=True)
        # ... adapt (points, pixels) pairs
        pairing_images = pairing_images[where]

        # Get point features
        pc = self.prepare_input_features(pc)

        # Projection on 2D grid
        cell_ind = self.get_occupied_2d_cells(pc)

        # Embedding
        kdtree = KDTree(pc[:, :3])
        assert pc.shape[0] > self.num_neighbors
        _, neighbors = kdtree.query(pc[:, :3], k=self.num_neighbors + 1)

        out = (
            pc[:, 3:].T[None],
            images,
            np.arange(pc.shape[0]),
            pairing_images,
            cell_ind[None],
            neighbors.T[None],
        )

        return out
