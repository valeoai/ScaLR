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

import warnings

import numpy as np


class WarmupCosine:
    def __init__(self, warmup_end, max_it, factor_min):
        self.max_it = max_it
        self.warmup_end = warmup_end
        self.factor_min = factor_min

    def __call__(self, it):
        if it < self.warmup_end:
            factor = it / self.warmup_end
        else:
            it = it - self.warmup_end
            max_it = self.max_it - self.warmup_end
            it = (it / max_it) * np.pi
            factor = self.factor_min + 0.5 * (1 - self.factor_min) * (np.cos(it) + 1)
        return factor


class LinWarmup_ReciprocalSqrt_LinCoolDown:
    def __init__(self, max_it, warmup_end=None, cooldown_start=None, timescale=None):
        self.max_it = max_it
        self.warmup_end = max_it // 10 if warmup_end is None else warmup_end
        self.cooldown_start = (
            max_it - 2 * (max_it // 10) if cooldown_start is None else cooldown_start
        )
        self.timescale = max_it // 100 if timescale is None else timescale
        assert self.warmup_end > 0
        assert self.warmup_end <= self.cooldown_start
        assert self.cooldown_start <= self.max_it

    def __call__(self, it):
        if it <= self.warmup_end:
            factor = it / self.warmup_end
        elif it > self.warmup_end and it <= self.cooldown_start:
            it = it - self.warmup_end
            factor = 1 / np.sqrt(1 + it / self.timescale)
        elif it > self.cooldown_start and it < self.max_it:
            it = it - self.cooldown_start
            max_it = self.max_it - self.cooldown_start
            base_value = 1 / np.sqrt(1 + self.cooldown_start / self.timescale)
            factor = base_value * (max_it - it - 1) / (max_it - 1)
        else:
            factor = 0
            warnings.warn(
                f"Number of iteration is {it} but max was set to {self.max_it}."
            )
        return factor
