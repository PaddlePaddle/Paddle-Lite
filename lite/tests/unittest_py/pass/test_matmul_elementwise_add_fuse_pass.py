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


class TestMatmulElementwiseAddFusePass(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        #opencl
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
        # metal
        metal_places = [
            Place(TargetType.Metal, PrecisionType.FP32,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.Metal, PrecisionType.FP16,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.ARM, PrecisionType.FP32),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        # self.enable_testing_on_place(places=metal_places)

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        matmul_x_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=30), min_size=2, max_size=4))
        transpose_Y_data = draw(st.sampled_from([False, True]))
        matmul_y_shape = []
        matmul_y1 = draw(st.integers(min_value=5, max_value=10))
        if transpose_Y_data == False:
            matmul_y_shape = [matmul_x_shape[-1], matmul_y1]
        else:
            matmul_y_shape = [matmul_y1, matmul_x_shape[-1]]
        add_x_data_shape = draw(
            st.sampled_from([[int(1), matmul_y1], [matmul_y1]]))

        matmul_op = OpConfig(
            type="matmul",
            inputs={"X": ["x_data"],
                    "Y": ["y_data"]},
            outputs={"Out": ["matmul_output_data"]},
            attrs={
                "transpose_X": False,
                "transpose_Y": transpose_Y_data,
                "alpha": 1.0,
                "fused_reshape_X": [],
                "fused_reshape_Y": [],
                "fused_transpose_X": [],
                "fused_transpose_Y": [],
                "fused_reshape_Out": [],
                "fused_transpose_Out": [],
                "head_number": int(1)
            })

        elementwise_add_op = OpConfig(
            type="elementwise_add",
            inputs={"X": ["matmul_output_data"],
                    "Y": ["add_x_data"]},
            outputs={"Out": ["output_data"]},
            attrs={"axis": -1})

        ops = [matmul_op, elementwise_add_op]

        weights_ = {
            "add_x_data": TensorConfig(shape=add_x_data_shape),
            "y_data": TensorConfig(shape=matmul_y_shape)
        }
        inputs_ = {"x_data": TensorConfig(shape=matmul_x_shape)}

        program_config = ProgramConfig(
            ops=ops, weights=weights_, inputs=inputs_,
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ['fc'], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["lite_matmul_elementwise_fuse_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
