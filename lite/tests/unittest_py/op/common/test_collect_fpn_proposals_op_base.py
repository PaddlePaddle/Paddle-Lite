# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
sys.path.append('..')

from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest
import hypothesis
import hypothesis.strategies as st


def sample_program_configs(draw):
    rois_shape = draw(st.sampled_from([[30, 4], [80, 4], [70, 4], [66, 4]]))
    scores_shape = draw(st.sampled_from([[30, 1], [65, 1], [70, 1]]))
    post_nms_topN = draw(st.integers(min_value=1, max_value=10))
    lod_data = [[1, 1, 1, 1]]

    def generate_rois(*args, **kwargs):
        return np.random.random(rois_shape).astype(np.float32)

    def generate_scores(*args, **kwargs):
        return np.random.random(scores_shape).astype(np.float32)

    collect_fpn_proposals_op = OpConfig(
        type="collect_fpn_proposals",
        inputs={
            "MultiLevelRois": ["multi_level_rois_data"],
            "MultiLevelScores": ["multi_level_scores_data"],
            "RoisNum": ["rois_num_data"]
        },
        outputs={
            "FpnRois": ["fpn_rois_data"],
            "MultiLevelRoIsNum": ["multi_level_rois_num_data"]
        },
        attrs={"post_nms_topN": post_nms_topN})
    program_config = ProgramConfig(
        ops=[collect_fpn_proposals_op],
        weights={},
        inputs={
            "multi_level_rois_data":
            TensorConfig(data_gen=partial(generate_rois)),
            "multi_level_scores_data":
            TensorConfig(data_gen=partial(generate_scores)),
            "rois_num_data": TensorConfig(data_gen=partial(generate_rois))
        },
        outputs=["fpn_rois_data", "multi_level_rois_num_data"])
    return program_config
