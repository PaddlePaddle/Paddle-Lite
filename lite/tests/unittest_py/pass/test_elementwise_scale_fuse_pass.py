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

from auto_scan_test import FusePassAutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
from test_elementwise_util import trim_trailing_singular_dims, check_input_shape_available
import hypothesis.strategies as st


class TestElementwiseScaleFuse(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
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

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        if len(program_config.inputs["input_data_x"].shape) > 4 or len(
                program_config.inputs["input_data_y"].shape
        ) > 4 or program_config.ops[1].attrs["bias_after_scale"] == False:
            return False
        return True

    def sample_program_configs(self, draw):
        in_shape_x = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=20), min_size=2, max_size=5))
        in_shape_y = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=20), min_size=2, max_size=5))

        axis = draw(
            st.integers(
                min_value=-1, max_value=max(len(in_shape_x), len(in_shape_y))))

        assume(
            check_input_shape_available(
                in_shape_x=in_shape_x, in_shape_y=in_shape_y, axis=axis) ==
            True)

        #scale param
        scale = draw(st.floats(min_value=0.5, max_value=5))
        bias = draw(st.floats(min_value=0, max_value=1))
        bias_after_scale = draw(st.sampled_from([False, True]))

        elementwise_op = OpConfig(
            type='elementwise_mul',
            inputs={"X": ["input_data_x"],
                    "Y": ["input_data_y"]},
            outputs={"Out": ["elementwise_output_data"]},
            attrs={"data_format": 'nchw',
                   "axis": axis})

        scale_op = OpConfig(
            type='scale',
            inputs={"X": ["elementwise_output_data"]},
            outputs={"Out": ["output_data"]},
            attrs={
                "scale": scale,
                "bias": bias,
                "bias_after_scale": bias_after_scale
            })

        ops = [elementwise_op, scale_op]
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input_data_x": TensorConfig(shape=in_shape_x),
                "input_data_y": TensorConfig(shape=in_shape_y)
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        config = CxxConfig()
        return self.get_predictor_configs(), ['elementwise_mul'], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(
            quant=False,
            max_examples=1000,
            passes=["lite_elementwise_scale_fuse_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
