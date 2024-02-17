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

from .pc_dataset import Collate
from .im_pc_dataset import CollateDistillation
from .merged_datasets import MergedDatasetsDistill
from .nuscenes_for_scalr import (
    NuScenesSemSeg,
    NuScenesSemSeg_1p,
    NuScenesSemSeg_10p,
    NuScenesDistill,
)
from .semantic_kitti_for_scalr import (
    SemanticKITTISemSeg,
    SemanticKITTISemSeg_1p,
    SemanticKITTIDistill,
)
from .pandaset_for_scalr import (
    Pandaset64SemSeg,
    PandasetGTSemSeg,
    PandaSet64Distill,
    PandaSetGTDistill,
)


LIST_DATASETS = {
    "nuscenes": NuScenesSemSeg,
    "nuscenes_1p": NuScenesSemSeg_1p,
    "nuscenes_10p": NuScenesSemSeg_10p,
    "semantic_kitti": SemanticKITTISemSeg,
    "semantic_kitti_1p": SemanticKITTISemSeg_1p,
    "panda64": Pandaset64SemSeg,
    "pandagt": PandasetGTSemSeg,
}

LIST_DATASETS_DISTILL = {
    "nuscenes": NuScenesDistill,
    "semantic_kitti": SemanticKITTIDistill,
    "panda64": PandaSet64Distill,
    "pandagt": PandaSetGTDistill,
    "merged_datasets": MergedDatasetsDistill,
}
