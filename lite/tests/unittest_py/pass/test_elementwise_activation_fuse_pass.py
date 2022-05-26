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


class TestElementwiseActivationFuse(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.X86, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])
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
        if predictor_config.target() == TargetType.OpenCL:
            input_shape_x = list(program_config.inputs["input_data_x"].shape)
            input_shape_y = list(program_config.inputs["input_data_y"].shape)
            if len(input_shape_x) > 4 or len(input_shape_y) > 4:
                return False
        return True

    def sample_program_configs(self, draw):

        elementwise_type = draw(
            st.sampled_from(
                ["elementwise_add", "elementwise_sub", "elementwise_mul"]))
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

        elementwise_op = OpConfig(
            type=elementwise_type,
            inputs={"X": ["input_data_x"],
                    "Y": ["input_data_y"]},
            outputs={"Out": ["elementwise_output_data"]},
            attrs={"data_format": 'nchw',
                   "axis": axis})

        target_str = self.get_target()
        act_list = ['relu']
        if target_str == "OpenCL":
            act_list.append("relu6")
            act_list.append("gelu")
        act_type = draw(st.sampled_from(act_list))

        def generate_act_attrs(act_type_str):
            attrs = {}
            if act_type_str == 'relu':
                attrs = {}
            return attrs

        active_op = OpConfig(
            type=act_type,
            inputs={"X": ["elementwise_output_data"]},
            outputs={"Out": ["output_data"]},
            attrs=generate_act_attrs(act_type))

        ops = [elementwise_op, active_op]
        self.ops = ops
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
        return self.get_predictor_configs(
        ), ["fusion_" + self.ops[0].type + "_activation"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(
            quant=False,
            max_examples=300,
            passes=["lite_elementwise_activation_fuse_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
