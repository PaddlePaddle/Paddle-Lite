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
'''
import sys
sys.path.append('../')

from auto_scan_test import AutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st
import argparse
from functools import partial
import random
import numpy as np


class TestCollectFpnProposalsOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        # it doesn't support std::vector<tensor>
        return False

    def sample_program_configs(self, draw):
        rois_shape = draw(
            st.sampled_from([[30, 4], [80, 4], [70, 4], [66, 4]]))
        scores_shape = draw(st.sampled_from([[30, 1], [65, 1], [70, 1]]))
        post_nms_topN = draw(st.integers(min_value=1, max_value=10))
        lod_data = [[1, 1, 1, 1]]

        def generate_rois(*args, **kwargs):
            return np.random.random(rois_shape).astype(np.float32)

        def generate_scores(*args, **kwargs):
            return np.random.random(scores_shape).astype(np.float32)

        def generate_rois_num(*args, **kwargs):
            return np.random.random(rois_shape).astype(np.int32)

        collect_fpn_proposals_op = OpConfig(
            type="collect_fpn_proposals",
            inputs={
                "MultiLevelRois": ["multi_level_rois_data"],
                "MultiLevelScores": ["multi_level_scores_data"],
                "MultiLevelRoIsNum": ["multi_level_rois_num_data"]
            },
            outputs={
                "FpnRois": ["fpn_rois_data"],
                "RoisNum": ["rois_num_data"]
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
                "multi_level_rois_num_data":
                TensorConfig(data_gen=partial(generate_rois_num))
            },
            outputs=["fpn_rois_data", "rois_num_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["collect_fpn_proposals"], (1e-5,
                                                                         1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
'''
