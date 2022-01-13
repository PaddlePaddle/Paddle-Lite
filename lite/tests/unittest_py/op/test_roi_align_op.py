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
sys.path.append('../')

from auto_scan_test import AutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import unittest
from functools import partial
import numpy as np
import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st
import argparse


class TestRoiAlignOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        # self.enable_testing_on_place(
        #     TargetType.Host,
        #     PrecisionType.FP32,
        #     DataLayoutType.NCHW,
        #     thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=32), min_size=4, max_size=4))
        spatial_scale = draw(st.floats(min_value=0.1, max_value=1.0))
        pooled_height = draw(st.integers(min_value=1, max_value=2))
        pooled_width = draw(st.integers(min_value=1, max_value=2))
        sampling_ratio = draw(st.sampled_from([-1, 4, 8]))
        aligned = draw(st.booleans())

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        def generate_rois(*args, **kwargs):
            return np.random.random([3, 4]).astype(np.float32)

        def generate_roisnum(*args, **kwargs):
            return np.random.random([in_shape[0]]).astype(np.int32)

        roi_align_op = OpConfig(
            type="roi_align",
            inputs={
                "X": ["input_data"],
                "ROIs": ["rois_data"],
                "RoisNum": ["roisnum_data"]
            },
            outputs={"Out": ["output_data"]},
            attrs={
                "spatial_scale": spatial_scale,
                "pooled_height": pooled_height,
                "pooled_width": pooled_width,
                "sampling_ratio": sampling_ratio,
                "aligned": aligned
            })
        program_config = ProgramConfig(
            ops=[roi_align_op],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input)),
                "rois_data": TensorConfig(data_gen=partial(generate_rois)),
                "roisnum_data":
                TensorConfig(data_gen=partial(generate_roisnum))
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["roi_align"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, min_success_num=25, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
