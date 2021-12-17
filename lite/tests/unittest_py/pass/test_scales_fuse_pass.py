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

from auto_scan_test import FusePassAutoScanTest
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class TestScalesFusePass(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        # opencl
        opencl_places = [
            Place(TargetType.OpenCL, PrecisionType.FP16,
                  DataLayoutType.ImageDefault), Place(
                      TargetType.OpenCL, PrecisionType.FP16,
                      DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
            Place(TargetType.OpenCL, PrecisionType.Any,
                  DataLayoutType.ImageDefault), Place(
                      TargetType.OpenCL, PrecisionType.Any,
                      DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(places=opencl_places)
        #x86
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape_x = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=64), min_size=2, max_size=4))
        scale1 = draw(st.floats(min_value=0.5, max_value=5))
        bias1 = draw(st.floats(min_value=0, max_value=1))
        scale2 = draw(st.floats(min_value=0.5, max_value=5))
        bias2 = draw(st.floats(min_value=0, max_value=1))
        bias_after_scale1 = draw(st.sampled_from([True]))  #required in pass
        bias_after_scale2 = draw(st.sampled_from([True]))  #required in pass

        scale1_op = OpConfig(
            type="scale",
            inputs={"X": ["input_data_x"]},
            outputs={"Out": ["scale1_output_data"]},
            attrs={
                "data_format": 'nchw',
                "scale": scale1,
                "bias": bias1,
                "bias_after_scale": bias_after_scale1
            })

        scale2_op = OpConfig(
            type="scale",
            inputs={"X": ["scale1_output_data"]},
            outputs={"Out": ["output_data"]},
            attrs={
                "scale": scale2,
                "bias": bias2,
                "bias_after_scale": bias_after_scale2
            })

        ops = [scale1_op, scale2_op]
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={"input_data_x": TensorConfig(shape=in_shape_x)},
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        if self.get_target() == 'OpenCL':
            return self.get_predictor_configs(
            ), ['io_copy', 'layout', 'scale', 'layout', 'io_copy'], (1e-5,
                                                                     1e-5)
        else:
            return self.get_predictor_configs(), ['scale'], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(
            quant=False, max_examples=25, passes=["lite_scales_fuse_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
