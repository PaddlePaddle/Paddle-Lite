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


class TestReshape2MatmulFusePass(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        #opencl
        # opencl_places = [
        #     Place(TargetType.OpenCL, PrecisionType.FP16,
        #           DataLayoutType.ImageDefault), Place(
        #               TargetType.OpenCL, PrecisionType.FP16,
        #               DataLayoutType.ImageFolder),
        #     Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
        #     Place(TargetType.OpenCL, PrecisionType.Any,
        #           DataLayoutType.ImageDefault), Place(
        #               TargetType.OpenCL, PrecisionType.Any,
        #               DataLayoutType.ImageFolder),
        #     Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
        #     Place(TargetType.Host, PrecisionType.FP32)
        # ]
        # self.enable_testing_on_place(places=opencl_places)
        #x86
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        # get input&output shape, get op attributes
        x_shape = list(program_config.inputs["input_data_x"].shape)
        y_shape = list(program_config.inputs["input_data_y"].shape)

        # if predictor_config.target() == TargetType.OpenCL:
        #     return False
        if x_shape[1] != y_shape[0]:
            return False
        return True

    def sample_program_configs(self, draw):
        alpha = draw(st.floats(min_value=1, max_value=1))
        x_num_col_dims = draw(st.integers(min_value=1, max_value=1))
        y_num_col_dims = draw(st.integers(min_value=1, max_value=1))
        n_int32 = draw(st.integers(min_value=1, max_value=64))
        c_int32 = draw(st.integers(min_value=1, max_value=64))
        matmul_y_w = draw(st.integers(min_value=1, max_value=64))

        dim2_values_h = draw(st.integers(min_value=1, max_value=64))
        dim2_values_w = draw(st.integers(min_value=1, max_value=64))
        assume(dim2_values_h * dim2_values_w == n_int32 * c_int32)

        dim4_values = [n_int32, c_int32, 1, 1]
        dim2_values = [dim2_values_h, dim2_values_w]
        matmul_y_shape = [dim2_values_w, matmul_y_w]

        reshape2_op = OpConfig(
            type="reshape2",
            inputs={"X": ["input_data_x"]},
            outputs={
                "Out": ["reshape2_output"],
                "XShape": ["reshape2_output_XShape"]
            },
            attrs={
                "shape": dim2_values  #compare input_data_x->shape
            })

        matmul_op = OpConfig(
            type="matmul",
            inputs={"X": ["reshape2_output"],
                    "Y": ["input_data_y"]},
            outputs={"Out": ["output_data"]},
            attrs={
                "transpose_X": False,
                "transpose_Y": False,
                "x_num_col_dims": x_num_col_dims,
                "y_num_col_dims": y_num_col_dims,
                "alpha": alpha,
                "fused_reshape_X": [],
                "fused_transpose_X": [],
                "fused_reshape_Y": [],
                "fused_transpose_Y": [],
                "fused_reshape_Out": [],
                "fused_transpose_Out": [],
                "head_number": int(1)
            })

        ops = [reshape2_op, matmul_op]
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input_data_x": TensorConfig(shape=dim4_values),
                "input_data_y": TensorConfig(shape=matmul_y_shape)
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ['mul'], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(
            quant=False,
            max_examples=300,
            min_success_num=25,
            passes=["lite_reshape2_matmul_fuse_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
