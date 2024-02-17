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

import os
import copy
import yaml
import numpy as np
from PIL import Image
from glob import glob
from .pc_dataset import PCDataset
from .im_pc_dataset import ImPcDataset

# For normalizing intensities
MEAN_INT = 0.28613698
STD_INT = 0.14090556


class SemanticKITTISemSeg(PCDataset):

    CLASS_NAME = [
        "car",  # 0
        "bicycle",  # 1
        "motorcycle",  # 2
        "truck",  # 3
        "other-vehicle",  # 4
        "person",  # 5
        "bicyclist",  # 6
        "motorcyclist",  # 7
        "road",  # 8
        "parking",  # 9
        "sidewalk",  # 10
        "other-ground",  # 11
        "building",  # 12
        "fence",  # 13
        "vegetation",  # 14
        "trunk",  # 15
        "terrain",  # 16
        "pole",  # 17
        "traffic-sign",  # 18
    ]

    def __init__(self, ratio="100p", **kwargs):
        super().__init__(**kwargs)

        # For normalizing intensities
        self.mean_int = MEAN_INT
        self.std_int = STD_INT

        # Config file and class mapping
        current_folder = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(current_folder, "semantic-kitti.yaml")) as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.mapper = np.vectorize(semkittiyaml["learning_map"].__getitem__)

        # Split
        if self.phase == "train":
            split = semkittiyaml["split"]["train"]
        elif self.phase == "val":
            split = semkittiyaml["split"]["valid"]
        elif self.phase == "test":
            split = semkittiyaml["split"]["test"]
        elif self.phase == "trainval":
            split = semkittiyaml["split"]["train"] + semkittiyaml["split"]["valid"]
        else:
            raise Exception(f"Unknown split {self.phase}")

        # Find all files
        self.im_idx = []
        for i_folder in np.sort(split):
            self.im_idx.extend(
                glob(
                    os.path.join(
                        self.rootdir,
                        "dataset",
                        "sequences",
                        str(i_folder).zfill(2),
                        "velodyne",
                        "*.bin",
                    )
                )
            )

        if self.phase == "train" and ratio != "100p":
            if ratio == "1p":
                skip_ratio = 100
            else:
                raise ValueError(f"Split {ratio} not coded")
            self.im_idx = sorted(self.im_idx)[::skip_ratio]
        else:
            print("Using original split")
            self.im_idx = np.sort(self.im_idx)

    def __len__(self):
        return len(self.im_idx)

    def load_pc(self, index):
        # Load point cloud
        pc = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))

        # Extract Label
        labels_inst = np.fromfile(
            self.im_idx[index].replace("velodyne", "labels")[:-3] + "label",
            dtype=np.uint32,
        ).reshape((-1, 1))
        labels = labels_inst & 0xFFFF  # delete high 16 digits binary
        labels = self.mapper(labels).astype(np.int32)

        # Map ignore index (0) to 255
        labels = labels[:, 0] - 1
        labels[labels == -1] = 255

        return pc, labels, self.im_idx[index]


class SemanticKITTIDistill(ImPcDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # For normalizing intensities
        self.mean_int = MEAN_INT
        self.std_int = STD_INT

        # Config file and class mapping
        current_folder = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(current_folder, "semantic-kitti.yaml")) as stream:
            semkittiyaml = yaml.safe_load(stream)

        # Split
        if self.phase == "train":
            split = semkittiyaml["split"]["train"]
        else:
            raise Exception(f"Unknown split {self.phase}")

        # Find all files
        self.im_idx = []
        for i_folder in np.sort(split):
            self.im_idx.extend(
                glob(
                    os.path.join(
                        self.rootdir,
                        "dataset",
                        "sequences",
                        str(i_folder).zfill(2),
                        "velodyne",
                        "*.bin",
                    )
                )
            )
        self.im_idx = np.sort(self.im_idx)
        assert len(self.im_idx) == 19130

    def __len__(self):
        return len(self.im_idx)

    def load_pc(self, index):
        pc = np.fromfile(self.im_idx[index], dtype=np.float32)
        return pc.reshape((-1, 4))

    def read_calibration(self, idx):
        """Calibration camera & point cloud"""
        #
        seq = os.path.split(self.im_idx[idx])[0]
        first = len(os.path.join(self.rootdir, "dataset", "sequences"))
        seq = seq[first + 1 : first + 4]
        calib_path = os.path.join(
            self.rootdir,
            "calib",
            "dataset",
            "sequences",
            seq,
            "calib.txt",
        )
        #
        calib_all = {}
        with open(calib_path) as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])
        # reshape matrices
        calib_out = {}
        calib_out["P2"] = calib_all["P2"].reshape(
            3, 4
        )  # 3x4 projection matrix for left camera
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        #
        return np.matmul(calib_out["P2"], calib_out["Tr"])

    def map_pc_to_image(self, pc, index, min_dist=1.0):
        # On SemanticKITTI only the points in the front are viewed
        pc_base = pc[pc[:, 0] > min_dist, :]

        # Load image
        im = np.array(
            Image.open(
                self.im_idx[index]
                .replace("velodyne", "image_2")
                .replace(".bin", ".png")
            )
        )

        # Read calibration matrice
        proj_matrix = self.read_calibration(index)

        # Project point on the camera
        pc_copy = copy.deepcopy(pc_base)
        projected_points = np.concatenate(
            (pc_copy[:, :3], np.ones((len(pc_copy), 1))), axis=1
        )
        projected_points = projected_points @ proj_matrix.T
        projected_points = projected_points[:, :2] / projected_points[:, 2:3]

        # Remove points that are either outside or behind the camera.
        # Also make sure points are at least 1m in front of the camera
        mask = np.ones(pc_copy.shape[0], dtype=bool)
        mask = np.logical_and(mask, projected_points[:, 0] > 0)
        mask = np.logical_and(mask, projected_points[:, 0] < im.shape[1] - 1)
        mask = np.logical_and(mask, projected_points[:, 1] > 0)
        mask = np.logical_and(mask, projected_points[:, 1] < im.shape[0] - 1)

        # Apply the mask
        projected_points = projected_points[mask]
        pc_base = pc_base[mask]

        # For points with a matching pixel, coordinates of that pixel (size N x 2)
        # Use flip for change from (x, y) to (row, column)
        matching_pixels = np.floor(np.flip(projected_points, axis=1)).astype(np.int64)

        # Append data
        images = [im / 255.0]
        matching_pixels = np.concatenate(
            (
                np.zeros((matching_pixels.shape[0], 1), dtype=np.int64),
                matching_pixels,
            ),
            axis=1,
        )
        pairing_images = [matching_pixels]

        return pc_base, images, np.concatenate(pairing_images)


class SemanticKITTISemSeg_1p(SemanticKITTISemSeg):
    def __init__(self, **kwargs):
        super().__init__(ratio="1p", **kwargs)
