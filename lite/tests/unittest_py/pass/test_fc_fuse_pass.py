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
import hypothesis.strategies as st


class TestFcFuse(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])
        #OpenCL not support mul 
        '''
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
        '''

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        if predictor_config.target() != TargetType.X86:
            return len(program_config.ops) == 2
        return True

    def sample_program_configs(self, draw):
        has_relu = draw(st.sampled_from([True, False]))
        mul_x_in_shape = draw(
            st.lists(
                st.integers(
                    min_value=5, max_value=10), min_size=2, max_size=5))
        x_num_col_dims_data = draw(
            st.integers(
                min_value=1, max_value=len(mul_x_in_shape) - 1))
        x0 = 1
        x1 = 1
        for i in range(0, x_num_col_dims_data):
            x0 = x0 * mul_x_in_shape[i]
        for i in range(x_num_col_dims_data, len(mul_x_in_shape)):
            x1 = x1 * mul_x_in_shape[i]

        #lite not check fuse condition : bias[0]=1 bias[1]=weight[1] 
        add_x_data_shape = draw(
            st.sampled_from([[
                1, draw(st.integers(
                    min_value=5, max_value=10))
            ], [draw(st.integers(
                min_value=5, max_value=10))]]))

        y_dims = 2
        y_num_col_dims = 1
        mul_out_shape = x_num_col_dims_data + y_dims - y_num_col_dims

        axis = mul_out_shape - len(add_x_data_shape)

        y1 = add_x_data_shape[0]
        if len(add_x_data_shape) == 2:
            y1 = add_x_data_shape[1]

        mul_op = OpConfig(
            type="mul",
            inputs={"X": ["mul_x_data"],
                    "Y": ["mul_y_data"]},
            outputs={"Out": ["mul_output_data"]},
            attrs={
                "x_num_col_dims": x_num_col_dims_data,
                "y_num_col_dims": 1
            })

        elementwise_add_op = OpConfig(
            type="elementwise_add",
            inputs={"X": ["mul_output_data"],
                    "Y": ["add_x_data"]},
            outputs={"Out": ["elementwise_add_output_data"]},
            attrs={"axis": axis})

        active_op = OpConfig(
            type="relu",
            inputs={"X": ["elementwise_add_output_data"]},
            outputs={"Out": ["output_data"]},
            attrs={})

        ops = [mul_op, elementwise_add_op]
        output_data = "elementwise_add_output_data"
        if has_relu:
            ops.append(active_op)
            output_data = "output_data"
        program_config = ProgramConfig(
            ops=ops,
            weights={"add_x_data": TensorConfig(shape=add_x_data_shape)},
            inputs={
                "mul_x_data": TensorConfig(shape=mul_x_in_shape),
                "mul_y_data": TensorConfig(shape=[x1, y1])
            },
            outputs=[output_data])
        return program_config

    def sample_predictor_configs(self):
        config = CxxConfig()
        return self.get_predictor_configs(), ['fc'], (1e-4, 1e-4)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(
            quant=False, max_examples=100, passes=["lite_fc_fuse_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
