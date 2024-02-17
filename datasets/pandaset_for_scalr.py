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
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
import transforms3d as t3d
from .pc_dataset import PCDataset
from .im_pc_dataset import ImPcDataset

MEAN_INT_64 = 18.23640649
STD_INT_64 = 25.86983417

MEAN_INT_GT = 23.06113565
STD_INT_GT = 16.76273235


def heading_position_to_mat(heading, position):
    """Original function is in pandaset devkit"""
    quat = np.array([heading["w"], heading["x"], heading["y"], heading["z"]])
    pos = np.array([position["x"], position["y"], position["z"]])
    transform_matrix = t3d.affines.compose(
        np.array(pos),
        t3d.quaternions.quat2mat(quat),
        [1.0, 1.0, 1.0],
    )
    return transform_matrix


def projection(
    lidar_points, camera_data, camera_pose, camera_intrinsics, filter_outliers=True
):
    """Original function is in pandaset devkit"""
    camera_heading = camera_pose["heading"]
    camera_position = camera_pose["position"]
    camera_pose_mat = heading_position_to_mat(camera_heading, camera_position)

    trans_lidar_to_camera = np.linalg.inv(camera_pose_mat)
    points3d_lidar = lidar_points
    points3d_camera = trans_lidar_to_camera[:3, :3] @ (
        points3d_lidar.T
    ) + trans_lidar_to_camera[:3, 3].reshape(3, 1)

    K = np.eye(3, dtype=np.float64)
    K[0, 0] = camera_intrinsics["fx"]
    K[1, 1] = camera_intrinsics["fy"]
    K[0, 2] = camera_intrinsics["cx"]
    K[1, 2] = camera_intrinsics["cy"]

    inliner_indices_arr = np.arange(points3d_camera.shape[1])
    if filter_outliers:
        condition = points3d_camera[2, :] > 0.0
        points3d_camera = points3d_camera[:, condition]
        inliner_indices_arr = inliner_indices_arr[condition]

    points2d_camera = K @ points3d_camera
    points2d_camera = (points2d_camera[:2, :] / points2d_camera[2, :]).T

    if filter_outliers:
        image_h, image_w = camera_data.shape[:2]
        condition = np.logical_and(
            (points2d_camera[:, 1] < image_h) & (points2d_camera[:, 1] > 0),
            (points2d_camera[:, 0] < image_w) & (points2d_camera[:, 0] > 0),
        )
        points2d_camera = points2d_camera[condition]
        points3d_camera = (points3d_camera.T)[condition]
        inliner_indices_arr = inliner_indices_arr[condition]
    return points2d_camera, points3d_camera, inliner_indices_arr


class PandasetSemSeg(PCDataset):

    MAPPING_CLASS = {
        0: 0,  # 'Noise' -> noise
        1: 0,  # 'Smoke' -> noise
        2: 0,  # 'Exhaust' -> noise
        3: 0,  # 'Spray or rain' -> noise
        4: 0,  # 'Reflection' -> noise
        5: 5,  # 'Vegetation' -> vegetation
        6: 16,  # 'Ground' -> ground
        7: 1,  # 'Road' -> road
        8: 6,  # 'Lane Line Marking' -> road marking
        9: 6,  # 'Stop Line Marking' -> road marking"
        10: 6,  # 'Other Road Marking' -> road marking"
        11: 7,  # 'Sidewalk' -> sidewalk
        12: 17,  # 'Driveway' -> driveway
        13: 10,  # 'Car' -> car
        14: 12,  # 'Pickup Truck' -> truck
        15: 12,  # 'Medium-sized Truck' -> truck
        16: 12,  # 'Semi-truck' -> truck
        17: 15,  # 'Towed Object' -> oth veh.
        18: 11,  # 'Motorcycle' -> motorcycle
        19: 15,  # 'Other Vehicle - Construction Vehicle' -> oth veh.
        20: 15,  # 'Other Vehicle - Uncommon' -> oth veh.
        21: 15,  # 'Other Vehicle - Pedicab' -> oth veh.
        22: 0,  # 'Emergency Vehicle' -> ignore
        23: 13,  # 'Bus' -> bus
        24: 0,  # 'Personal Mobility Device' -> ignore
        25: 0,  # 'Motorized Scooter' -> ignore
        26: 14,  # 'Bicycle' -> bicycle
        27: 15,  # 'Train' -> oth veh.
        28: 15,  # 'Trolley' -> oth veh.
        29: 15,  # 'Tram / Subway' -> oth veh.
        30: 4,  # 'Pedestrian' -> pedestrian
        31: 4,  # 'Pedestrian with Object' -> pedestrian
        32: 0,  # 'Animals - Bird' -> noise
        33: 0,  # 'Animals - Other'-> noise
        34: 9,  # 'Pylons' -> cones
        35: 3,  # 'Road Barriers' -> barrier
        36: 2,  # 'Signs' -> traffic sign
        37: 9,  # 'Cones' -> cones
        38: 2,  # 'Construction Signs' -> traffic sign
        39: 3,  # 'Temporary Construction Barriers' -> barrier
        40: 0,  # 'Rolling Containers', -> noise
        41: 8,  # 'Building' -> manmade
        42: 8,  # 'Other Static Object' -> manmade
    }

    CLASS_NAME = [
        "road",  # 1
        "traffic sign",  # 2
        "barrier",  # 3
        "pedestrian",  # 4
        "vegetation",  # 5
        "road marking",  # 6
        "sidewalk",  # 7
        "manmade",  # 8
        "traffic cone",  # 9
        "car",  # 10
        "motorcycle",  # 11
        "truck",  # 12
        "bus",  # 13
        "bicycle",  # 14
        "oth. vehicle",  # 15
        "ground",  # 16
        "driveway",  # 17
    ]

    def __init__(self, which_pandar=None, **kwargs):
        super().__init__(**kwargs)

        # Select lidar
        assert which_pandar in ["pandar_64", "pandar_gt"]
        self.which_pandar = 0 if which_pandar == "pandar_64" else 1

        # Class mapping
        self.mapping = np.vectorize(PandasetSemSeg.MAPPING_CLASS.__getitem__)

        # List of scenes
        scene_list = np.sort(glob(self.rootdir + "/*/annotations/semseg/"))
        scene_list = np.array([f.split("/")[-4] for f in scene_list])

        # GPS coordinates
        pos_scene = []
        for scene in scene_list:
            file_pose = self.rootdir + f"/{scene}/meta/gps.json"
            with open(file_pose, "r") as f:
                gps = json.load(f)
            pos_scene.append(gps[0]["lat"])
        pos_scene = np.array(pos_scene)

        # Split
        val_split = pos_scene <= 37.6
        train_split = pos_scene > 37.6
        if self.phase == "train":
            scene_list = scene_list[train_split]
            assert len(scene_list) == 49
        elif self.phase == "val":
            scene_list = scene_list[val_split]
            assert len(scene_list) == 27
        else:
            raise Exception(f"Unknown split {self.phase}")
        scene_list = np.sort(scene_list)

        # Find all files
        self.im_idx, self.poses = [], []
        for scene_id in scene_list:
            self.im_idx.extend(
                np.sort(glob(self.rootdir + f"/{scene_id}/lidar/*.pkl.gz"))
            )
            file_pose = self.rootdir + f"/{scene_id}/lidar/poses.json"
            with open(file_pose, "r") as f:
                self.poses.extend(json.load(f))

    def __len__(self):
        return len(self.im_idx)

    def load_pc(self, index):

        # Load pc and labels
        pc = pd.read_pickle(self.im_idx[index]).values
        where_pandar = pc[:, -1] == self.which_pandar
        pc = pc[where_pandar, :-2]
        assert pc.shape[1] == 4

        # Load label
        label = pd.read_pickle(
            self.im_idx[index].replace("lidar", "annotations/semseg")
        ).values
        label = label[where_pandar, 0]
        label = self.mapping(label) - 1
        label[label == -1] = 255

        # Transform to ego coordinate system
        pose = self.poses[index]
        transform_matrix = heading_position_to_mat(pose["heading"], pose["position"])
        transform_matrix = np.linalg.inv(transform_matrix)
        pc[:, :3] = pc[:, :3] @ transform_matrix[:3, :3].T
        pc[:, :3] += transform_matrix[:3, [3]].T

        # Extra shift along z-axis to have road approximately 1.7 m below the center of coord system
        pc[:, 2] -= 1.6

        return pc, label, self.im_idx[index]


class PandasetDistill(ImPcDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # List of scenes
        scene_list = np.sort(glob(self.rootdir + "/*/annotations/semseg/"))
        scene_list = np.array([f.split("/")[-4] for f in scene_list])

        # List of available cameras
        self.camera_list = [
            "back_camera",
            "front_camera",
            "front_left_camera",
            "front_right_camera",
            "left_camera",
            "right_camera",
        ]

        # GPS coordinates
        pos_scene = []
        for scene in scene_list:
            file_pose = self.rootdir + f"/{scene}/meta/gps.json"
            with open(file_pose, "r") as f:
                gps = json.load(f)
            pos_scene.append(gps[0]["lat"])
        pos_scene = np.array(pos_scene)

        # Split
        train_split = pos_scene > 37.6
        scene_list = scene_list[train_split]
        scene_list = np.sort(scene_list)
        assert len(scene_list) == 49

        # Find all point clouds
        self.im_idx, self.pc_poses = [], []
        for scene_id in scene_list:
            self.im_idx.extend(
                np.sort(glob(self.rootdir + f"/{scene_id}/lidar/*.pkl.gz"))
            )
            file_pose = self.rootdir + f"/{scene_id}/lidar/poses.json"
            with open(file_pose, "r") as f:
                self.pc_poses.extend(json.load(f))

    def __len__(self):
        return len(self.im_idx)

    def load_pc(self, index):
        pc_base = pd.read_pickle(self.im_idx[index]).values
        where_pandar = pc_base[:, -1] == self.which_pandar
        pc_base = pc_base[where_pandar, :-2]
        assert pc_base.shape[1] == 4
        return pc_base

    def map_pc_to_image(self, pc, index, min_dist=1.0):
        # Reference point cloud
        pc_base = pc

        # Choose one camera
        if self.which_pandar == 0:
            # Pandar64 has FOV of 360 deg, so pick any camera
            camera_name = self.camera_list[
                torch.randint(len(self.camera_list), (1,))[0]
            ]
        else:
            # PandarGT is only sampling on the front
            camera_name = "front_camera"
        path_camera, id_pc = os.path.split(
            self.im_idx[index].replace("lidar", f"camera/{camera_name}/")
        )
        id_pc = id_pc.split(".")[0]

        # Load image
        im = np.array(Image.open(path_camera + f"/{id_pc}.jpg"))

        # Read calibration matrix
        file_pose = path_camera + "/poses.json"
        with open(file_pose, "r") as f:
            camera_pose = json.load(f)
        file_pose = path_camera + "/intrinsics.json"
        with open(file_pose, "r") as f:
            camera_intrinsics = json.load(f)

        # Project points to camera
        matching_pixels, points3d_camera, inliner_indices_arr = projection(
            copy.deepcopy(pc_base[:, :3]),
            im,
            camera_pose[int(id_pc)],
            camera_intrinsics,
            filter_outliers=True,
        )
        matching_pixels = np.floor(np.fliplr(matching_pixels)).astype(np.int64)

        # Keep only points viewed in camera
        pc_base = pc_base[inliner_indices_arr]

        # Transform point cloud to ego coordinate system
        pose = self.pc_poses[index]
        transform_matrix = heading_position_to_mat(pose["heading"], pose["position"])
        transform_matrix = np.linalg.inv(transform_matrix)
        pc_base[:, :3] = pc_base[:, :3] @ transform_matrix[:3, :3].T
        pc_base[:, :3] += transform_matrix[:3, [3]].T
        # Extra shift along z-axis to have road approximately 1.7 m below the center of coord system
        pc_base[:, 2] -= 1.6

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


class Pandaset64SemSeg(PandasetSemSeg):
    def __init__(self, **kwargs):
        super().__init__(which_pandar="pandar_64", **kwargs)

        self.mean_int = MEAN_INT_64
        self.std_int = STD_INT_64


class PandasetGTSemSeg(PandasetSemSeg):
    def __init__(self, **kwargs):
        super().__init__(which_pandar="pandar_gt", **kwargs)

        self.mean_int = MEAN_INT_GT
        self.std_int = STD_INT_GT


class PandaSet64Distill(PandasetDistill):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mean_int = MEAN_INT_64
        self.std_int = STD_INT_64
        self.which_pandar = 0


class PandaSetGTDistill(PandasetDistill):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mean_int = MEAN_INT_GT
        self.std_int = STD_INT_GT
        self.which_pandar = 1
