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
        return True

    def sample_program_configs(self, draw):
        act_type = draw(st.sampled_from(["", "relu", "relu6"]))
        op_type = draw(st.sampled_from(["mul", "matmul", "matmul_v2"]))
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
        inputs_data = {
            "mul_x_data": TensorConfig(shape=mul_x_in_shape),
            "mul_y_data": TensorConfig(shape=[x1, y1])
        }
        if op_type == "matmul" or op_type == "matmul_v2":
            if op_type == "matmul_v2":
                attrs_op = {
                    "trans_x": False,
                    "trans_y": False,
                }
            else:
                attrs_op = {
                    "transpose_X": False,
                    "transpose_Y": False,
                    "alpha": 1.0,
                    "fused_reshape_X": [],
                    "fused_reshape_Y": [],
                    "fused_transpose_X": [],
                    "fused_transpose_Y": [],
                    "fused_reshape_Out": [],
                    "fused_transpose_Out": [],
                    "Scale_x": 0.1,
                    "Scale_y": 0.1,
                    "Scale_out": 0.1,
                    "head_number": 1,
                    "force_fp32_output": False
                }
            mul_op = OpConfig(
                type=op_type,
                inputs={"X": ["mul_x_data"],
                        "Y": ["mul_y_data"]},
                outputs={"Out": ["mul_output_data"]},
                attrs=attrs_op)
            inputs_data = {
                "mul_x_data": TensorConfig(
                    shape=[draw(st.integers(
                        min_value=2, max_value=100)), x1]),
                "mul_y_data": TensorConfig(shape=[x1, y1])
            }
            axis = 2 - len(add_x_data_shape)

        elementwise_add_op = OpConfig(
            type="elementwise_add",
            inputs={"X": ["mul_output_data"],
                    "Y": ["add_x_data"]},
            outputs={"Out": ["elementwise_add_output_data"]},
            attrs={"axis": axis})

        act_attrs = {}
        if act_type == "relu6":
            act_attrs = {"threshold": 6.0, }
        active_op = OpConfig(
            type=act_type,
            inputs={"X": ["elementwise_add_output_data"]},
            outputs={"Out": ["output_data"]},
            attrs=act_attrs)

        ops = [mul_op, elementwise_add_op]
        output_data = "elementwise_add_output_data"
        if act_type == "relu" or act_type == "relu6":
            ops.append(active_op)
            output_data = "output_data"
        program_config = ProgramConfig(
            ops=ops,
            weights={"add_x_data": TensorConfig(shape=add_x_data_shape)},
            inputs=inputs_data,
            outputs=[output_data])
        return program_config

    def sample_predictor_configs(self):
        config = CxxConfig()
        return self.get_predictor_configs(), ['fc'], (1e-4, 1e-4)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            target_type = predictor_config.target()
            op_type = program_config.ops[0].type
            if target_type == TargetType.X86:
                if op_type == "matmul" or op_type == "matmul_v2":
                    return True
                if len(program_config.ops) > 2:
                    act_type = program_config.ops[2].type
                    if act_type == "relu6":
                        return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support this op/pass in a specific case on X86. We need to fix it as soon as possible."
        )

    def test(self, *args, **kwargs):
        self.run_and_statis(
            quant=False, max_examples=100, passes=["lite_fc_fuse_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
