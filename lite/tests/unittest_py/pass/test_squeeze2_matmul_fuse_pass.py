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


class TestSqueeze2MatmulFusePass(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        #x86
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        #Metal
        metal_places = [
            Place(TargetType.Metal, PrecisionType.FP32,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.Metal, PrecisionType.FP16,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.ARM, PrecisionType.FP32),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(places=metal_places)

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        target_type = predictor_config.target()
        in_shape = list(program_config.inputs["squeeze2_input_x"].shape)
        if target_type in [TargetType.Metal]:
            if in_shape[1] != 1:
                return False
        return True

    def sample_program_configs(self, draw):
        alpha = draw(st.floats(min_value=1, max_value=1))  #required in pass
        x_num_col_dims = draw(st.floats(min_value=0, max_value=1))
        y_num_col_dims = draw(st.floats(min_value=0, max_value=1))
        int32_values_1 = draw(st.integers(min_value=1, max_value=40))
        int32_values_2 = draw(st.integers(min_value=1, max_value=40))
        int32_values_3 = draw(st.integers(min_value=1, max_value=40))

        squeeze2_input_shape = [int32_values_1, int32_values_2, 1, 1]
        matmul_input_shape = [squeeze2_input_shape[1], int32_values_3]
        scale_x = draw(st.sampled_from([0.1, 1.1]))
        scale_y = draw(st.sampled_from([0.1, 1.1]))
        scale_out = draw(st.sampled_from([0.1, 1.1]))
        force_fp32_output = draw(st.booleans())
        squeeze2_op = OpConfig(
            type="squeeze2",
            inputs={"X": ["squeeze2_input_x"]},
            outputs={
                "Out": ["squeeze2_output"],
                "XShape": ["squeeze2_output_XShape"]
            },
            attrs={
                "axes": [2, 3]  #required in pass
            })

        matmul_op = OpConfig(
            type="matmul",
            inputs={"X": ["squeeze2_output"],
                    "Y": ["matmul_input"]},
            outputs={"Out": ["output_data"]},
            attrs={
                "transpose_X": False,  #required in pass
                "transpose_Y": False,  #required in pass
                "x_num_col_dims": x_num_col_dims,
                "y_num_col_dims": y_num_col_dims,
                "Scale_x": scale_x,
                "Scale_y": scale_y,
                "Scale_out": scale_out,
                "force_fp32_output": force_fp32_output,
                "alpha": alpha,
                "fused_reshape_X": [],
                "fused_transpose_X": [],
                "fused_reshape_Y": [],
                "fused_transpose_Y": [],
                "fused_reshape_Out": [],
                "fused_transpose_Out": [],
                "head_number": int(1)
            })

        ops = [squeeze2_op, matmul_op]
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "squeeze2_input_x": TensorConfig(shape=squeeze2_input_shape),
                "matmul_input": TensorConfig(shape=matmul_input_shape)
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        config_lists = self.get_predictor_configs()
        for config in config_lists:
            if config.target() in [TargetType.Metal]:
                atol, rtol = 1e-2, 1e-2

        return self.get_predictor_configs(), ["mul"], (atol, rtol)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 25
        if target_str in ["Metal"]:
            # Make sure to generate enough valid cases for specific targets
            max_examples = 500
        self.run_and_statis(
            quant=False,
            max_examples=max_examples,
            passes=["lite_squeeze2_matmul_fuse_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
