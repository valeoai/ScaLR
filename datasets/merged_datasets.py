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

from .im_pc_dataset import ImPcDataset
from .nuscenes_for_scalr import NuScenesDistill
from .semantic_kitti_for_scalr import SemanticKITTIDistill
from .pandaset_for_scalr import PandaSet64Distill, PandaSetGTDistill


class MergedDatasetsDistill(ImPcDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        rootdir = kwargs["rootdir"]

        kwargs["rootdir"] = rootdir + "/nuscenes/"
        self.nusc = NuScenesDistill(**kwargs)

        kwargs["rootdir"] = rootdir + "/semantic_kitti/"
        self.kitti = SemanticKITTIDistill(**kwargs)

        kwargs["rootdir"] = rootdir + "/pandaset/"
        self.pd_64 = PandaSet64Distill(**kwargs)
        self.pd_gt = PandaSetGTDistill(**kwargs)

        assert len(self) == 55100

    def __len__(self):
        return len(self.nusc) + len(self.kitti) + len(self.pd_64) + len(self.pd_gt)

    def __getitem__(self, idx):
        if idx < len(self.nusc):
            return self.nusc[idx]
        elif idx >= len(self.nusc) and idx < len(self.nusc) + len(self.kitti):
            return self.kitti[idx - len(self.nusc)]
        elif idx >= len(self.nusc) + len(self.kitti) and idx < len(self.nusc) + len(
            self.kitti
        ) + len(self.pd_64):
            return self.pd_64[idx - len(self.nusc) - len(self.kitti)]
        else:
            return self.pd_gt[idx - len(self.nusc) - len(self.kitti) - len(self.pd_64)]
