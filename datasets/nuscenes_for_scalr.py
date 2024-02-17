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
import torch
import numpy as np
from PIL import Image
from .pc_dataset import PCDataset
from pyquaternion import Quaternion
from .im_pc_dataset import ImPcDataset
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import LidarPointCloud

# For normalizing intensities
MEAN_INT = 18.742355
STD_INT = 22.04632


class ClassMapper:
    def __init__(self):
        current_folder = os.path.dirname(os.path.realpath(__file__))
        self.mapping = np.load(
            os.path.join(current_folder, "mapping_class_index_nuscenes.npy")
        )

    def get_index(self, x):
        return self.mapping[x] if x < len(self.mapping) else 0


class NuScenesSemSeg(PCDataset):

    CLASS_NAME = [
        "barrier",
        "bicycle",
        "bus",
        "car",
        "construction_vehicle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "trailer",
        "truck",
        "driveable_surface",
        "other_flat",
        "sidewalk",
        "terrain",
        "manmade",
        "vegetation",
    ]

    def __init__(self, ratio="100p", **kwargs):
        super().__init__(**kwargs)

        # For normalizing intensities
        self.mean_int = MEAN_INT
        self.std_int = STD_INT

        # Class mapping
        current_folder = os.path.dirname(os.path.realpath(__file__))
        self.mapper = np.vectorize(ClassMapper().get_index)

        # List all keyframes
        self.ratio = ratio
        if self.phase == "train":
            if self.ratio == "100p":
                self.list_frames = np.load(
                    os.path.join(current_folder, "list_files_nuscenes.npz")
                )[self.phase]
            elif self.ratio == "10p":
                self.list_frames = np.load(
                    os.path.join(current_folder, "nuscenes-ratio_10-v_0.npy"),
                    allow_pickle=True,
                )
            elif self.ratio == "1p":
                self.list_frames = np.load(
                    os.path.join(current_folder, "nuscenes-ratio_100-v_0.npy"),
                    allow_pickle=True,
                )
            else:
                raise ValueError(f"Unprepared nuScenes split {self.ratio}.")
        elif self.phase == "val":
            self.list_frames = np.load(
                os.path.join(current_folder, "list_files_nuscenes.npz")
            )[self.phase]

    def __len__(self):
        return len(self.list_frames)

    def load_pc(self, index):
        # Load point cloud
        pc = np.fromfile(
            os.path.join(self.rootdir, self.list_frames[index][0]),
            dtype=np.float32,
        )
        pc = pc.reshape((-1, 5))[:, :4]

        # Load segmentation labels
        labels = np.fromfile(
            os.path.join(self.rootdir, self.list_frames[index][1]),
            dtype=np.uint8,
        )
        labels = self.mapper(labels)

        # Label 0 should be ignored
        labels = labels - 1
        labels[labels == -1] = 255

        return pc, labels, self.list_frames[index][2]


class NuScenesDistill(ImPcDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # For normalizing intensities
        self.mean_int = MEAN_INT
        self.std_int = STD_INT

        # List of available cameras
        self.camera_list = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]

        # Load data
        self.list_keyframes = np.load(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                f"nuscenes_data_{self.phase}.npy",
            ),
            allow_pickle=True,
        ).item()
        assert len(self.list_keyframes) == 28130

    def __len__(self):
        return len(self.list_keyframes)

    def load_pc(self, index):
        pc = np.fromfile(
            os.path.join(
                self.rootdir,
                self.list_keyframes[index]["point"]["filename"],
            ),
            dtype=np.float32,
        )
        return pc.reshape((-1, 5))[:, :4]

    def map_pc_to_image(self, pc, index, min_dist=1.0):
        # Make point cloud compatible with nuscenes-devkit format
        pc_base = LidarPointCloud(pc.T)

        # Choose one camera
        camera_name = self.camera_list[torch.randint(len(self.camera_list), (1,))[0]]

        # Load image
        im = np.array(
            Image.open(
                os.path.join(
                    self.rootdir, self.list_keyframes[index][camera_name]["filename"]
                )
            )
        )

        # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        pc_copy = copy.deepcopy(pc_base)
        cs_record = self.list_keyframes[index]["point"]["cs_record"]
        pc_copy.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
        pc_copy.translate(np.array(cs_record["translation"]))

        # Second step: transform from ego to the global frame.
        poserecord = self.list_keyframes[index]["point"]["poserecord"]
        pc_copy.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
        pc_copy.translate(np.array(poserecord["translation"]))

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        poserecord = self.list_keyframes[index][camera_name]["poserecord"]
        pc_copy.translate(-np.array(poserecord["translation"]))
        pc_copy.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = self.list_keyframes[index][camera_name]["cs_record"]
        pc_copy.translate(-np.array(cs_record["translation"]))
        pc_copy.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc_copy.points[2, :]

        # Take a "picture" of the point cloud
        # (matrix multiplication with camera-matrix + renormalization).
        projected_points = view_points(
            pc_copy.points[:3, :],
            np.array(cs_record["camera_intrinsic"]),
            normalize=True,
        )

        # Remove points that are either outside or behind the camera.
        # Also make sure points are at least 1m in front of the camera
        projected_points = projected_points[:2].T
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, projected_points[:, 0] > 0)
        mask = np.logical_and(mask, projected_points[:, 0] < im.shape[1] - 1)
        mask = np.logical_and(mask, projected_points[:, 1] > 0)
        mask = np.logical_and(mask, projected_points[:, 1] < im.shape[0] - 1)

        # Apply the mask
        projected_points = projected_points[mask]
        pc_base.points = pc_base.points[:, mask]

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

        return pc_base.points.T, images, np.concatenate(pairing_images)


class NuScenesSemSeg_1p(NuScenesSemSeg):
    def __init__(self, **kwargs):
        super().__init__(ratio="1p", **kwargs)


class NuScenesSemSeg_10p(NuScenesSemSeg):
    def __init__(self, **kwargs):
        super().__init__(ratio="10p", **kwargs)
