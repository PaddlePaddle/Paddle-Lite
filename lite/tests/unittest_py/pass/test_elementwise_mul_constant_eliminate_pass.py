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


class TestElementwiseMulConstantEliminateFuse(FusePassAutoScanTest):
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
            fill_constant_shape = program_config.ops[1].attrs["shape"]
            input_shape = list(program_config.inputs["input_data"].shape)
            if len(fill_constant_shape) > 4 or len(input_shape) > 4:
                return False
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=20), min_size=2, max_size=5))
        fill_constant_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=20), min_size=2, max_size=5))

        axis = draw(
            st.integers(
                min_value=-1,
                max_value=max(len(in_shape), len(fill_constant_shape))))

        out_shape = []
        assume(
            check_input_shape_available(
                in_shape_x=in_shape,
                in_shape_y=fill_constant_shape,
                axis=axis,
                out_shape=out_shape) == True)
        assume(out_shape == in_shape)

        threshold = draw(st.floats(min_value=0, max_value=1))
        scale = draw(st.floats(min_value=0.5, max_value=5))
        offset = draw(st.floats(min_value=0, max_value=1))

        hard_swish_op0 = OpConfig(
            type="hard_swish",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["hard_swish_output_data"]},
            attrs={"threshold": threshold,
                   "scale": scale,
                   "offset": offset})

        fill_constant_op = OpConfig(
            type="fill_constant",
            inputs={},
            outputs={"Out": ["fill_constant_output_data"]},
            attrs={
                "dtype": 5,
                "shape": fill_constant_shape,
                "value": 1.,
                "force_cpu": False,
                "place_type": -1
            })

        elementwise_mul_op = OpConfig(
            type="elementwise_mul",
            inputs={
                "X": ["hard_swish_output_data"],
                "Y": ["fill_constant_output_data"]
            },
            outputs={"Out": ["elementwise_mul_output_data"]},
            attrs={"axis": axis})

        hard_swish_op1 = OpConfig(
            type="hard_swish",
            inputs={"X": ["elementwise_mul_output_data"]},
            outputs={"Out": ["output_data"]},
            attrs={"threshold": threshold,
                   "scale": scale,
                   "offset": offset})

        ops = [
            hard_swish_op0, fill_constant_op, elementwise_mul_op,
            hard_swish_op1
        ]
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={"input_data": TensorConfig(shape=in_shape)},
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        config = CxxConfig()
        return self.get_predictor_configs(), ['hard_swish', 'hard_swish'], (
            1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 500
        if target_str == "OpenCL":
            max_examples = 2000
        self.run_and_statis(
            quant=False,
            max_examples=max_examples,
            passes=["lite_elementwise_mul_constant_eliminate_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
