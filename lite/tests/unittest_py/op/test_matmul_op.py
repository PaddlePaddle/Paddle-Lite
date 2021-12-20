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

import hypothesis
from hypothesis import given, settings, seed, example, assume

import numpy as np
from functools import partial
import hypothesis.strategies as st


class TestMulOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(TargetType.ARM, PrecisionType.FP32,
                                     DataLayoutType.NCHW)
        self.enable_testing_on_place(TargetType.X86, PrecisionType.FP32,
                                     DataLayoutType.NCHW)
        self.enable_testing_on_place(TargetType.Metal, PrecisionType.FP32,
                                     DataLayoutType.NCHW)
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
        target_str = self.get_target()
        # opencl bugs to be fix in the future
        if target_str == "OpenCL":
            return False
            in_shape = list(program_config.inputs["input_data_x"].shape)
            if in_shape[0] % 4 != 0:
                return False
            if program_config.ops[0].attrs["alpha"] != 1.0:
                return False
        if target_str == "Metal":
            return False

        return True  # True ci run matmul has diff

    def sample_program_configs(self, draw):
        target_str = self.get_target()
        if target_str == "OpenCL":
            shape0 = draw(st.integers(min_value=1, max_value=4)) * 4
            shape1 = draw(st.integers(min_value=1, max_value=4)) * 4
            shape2 = draw(st.integers(min_value=1, max_value=4)) * 4
            batch0 = draw(st.integers(min_value=1, max_value=4)) * 4
            batch1 = draw(st.integers(min_value=1, max_value=4)) * 4
        if target_str == "ARM":
            shape0 = draw(st.integers(min_value=1, max_value=64))
            shape1 = draw(st.integers(min_value=1, max_value=64))
            shape2 = draw(st.integers(min_value=1, max_value=64))
            batch0 = draw(st.integers(min_value=1, max_value=64))
            batch1 = draw(st.integers(min_value=1, max_value=64))
        transpose_X = draw(st.booleans())
        transpose_Y = draw(st.booleans())
        if ((not transpose_X) and (not transpose_Y)):
            X_shape = [batch0, 1, shape0, shape1]
            Y_shape = [batch0, shape1, shape2]
        if ((transpose_X) and (not transpose_Y)):
            X_shape = [batch1, 1, shape1, shape0]
            Y_shape = [batch1, 1, shape1, shape2]
        if ((not transpose_X) and (transpose_Y)):
            X_shape = [batch0, shape0, shape1]
            Y_shape = [batch0, shape2, shape1]
        if ((transpose_X) and (transpose_Y)):
            X_shape = [batch0, shape1, shape0]
            Y_shape = [batch0, shape2, shape1]
        alpha = draw(st.sampled_from([0.1, 1.0, 1.1, -1.5]))
        fused_reshape_X = draw(st.sampled_from([[]]))
        fused_reshape_Y = draw(st.sampled_from([[]]))
        fused_transpose_X = draw(st.sampled_from([[]]))
        fused_transpose_Y = draw(st.sampled_from([[]]))
        fused_reshape_Out = draw(st.sampled_from([[]]))
        fused_transpose_Out = draw(st.sampled_from([[]]))
        Scale_x = draw(st.floats(min_value=0.1, max_value=10.0))
        Scale_y = draw(st.floats(min_value=0.1, max_value=10.0))
        Scale_out = draw(st.floats(min_value=0.1, max_value=10.0))
        force_fp32_output = draw(st.booleans())

        matmul_op = OpConfig(
            type="matmul",
            inputs={"X": ["input_data_x"],
                    "Y": ["input_data_y"]},
            outputs={"Out": ["output_data"]},
            attrs={
                "transpose_X": transpose_X,
                "transpose_Y": transpose_Y,
                "alpha": alpha,
                "fused_reshape_X": fused_reshape_X,
                "fused_reshape_Y": fused_reshape_Y,
                "fused_transpose_X": fused_transpose_X,
                "fused_transpose_Y": fused_transpose_Y,
                "fused_reshape_Out": fused_reshape_Out,
                "fused_transpose_Out": fused_transpose_Out,
                "Scale_x": Scale_x,
                "Scale_y": Scale_y,
                "Scale_out": Scale_out,
                "force_fp32_output": force_fp32_output
            })
        program_config = ProgramConfig(
            ops=[matmul_op],
            weights={},
            inputs={
                "input_data_x": TensorConfig(shape=X_shape),
                "input_data_y": TensorConfig(shape=Y_shape)
            },
            outputs={"output_data"})
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["matmul"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        sample_size = 25
        target_str = self.get_target()
        if target_str == "OpenCL":
            sample_size = 100
        self.run_and_statis(quant=False, max_examples=sample_size)


if __name__ == "__main__":
    unittest.main(argv=[''])
