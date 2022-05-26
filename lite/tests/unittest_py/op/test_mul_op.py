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
import hypothesis.strategies as st
import numpy as np


class TestMulOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP16,
            DataLayoutType.NCHW,
            thread=[1, 4])
        arm_valid_places = [
            Place(TargetType.ARM, PrecisionType.INT8, DataLayoutType.NCHW),
            Place(TargetType.ARM, PrecisionType.FP32, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=arm_valid_places, thread=[1, 4])

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
        shape0 = draw(st.integers(min_value=1, max_value=32))
        shape1 = draw(st.integers(min_value=1, max_value=32))
        shape2 = draw(st.integers(min_value=1, max_value=16))
        shape3 = draw(st.integers(min_value=1, max_value=16))
        shape4 = draw(st.integers(min_value=1, max_value=16))
        shape5 = draw(st.integers(min_value=1, max_value=16))
        shape6 = shape4 * shape5
        case = draw(st.sampled_from([1, 2, 3, 4]))
        if case == 1:
            x_num_col_dims = 2
            y_num_col_dims = 1
            shape3 = shape6
            shape2 = shape3
            X_shape = draw(
                st.sampled_from([[shape0, shape5, shape2],
                                 [shape0, shape5, shape4, shape5]]))
            Y_shape = draw(
                st.sampled_from([[shape3, shape4, shape5], [shape3, shape5]]))
        if case == 2:
            x_num_col_dims = 2
            y_num_col_dims = 2
            X_shape = draw(
                st.sampled_from([[shape0, shape1, shape6],
                                 [shape4, shape5, shape5, shape4]]))
            Y_shape = draw(st.sampled_from([[shape4, shape5, shape5]]))
        if case == 3:
            x_num_col_dims = 1
            y_num_col_dims = 1
            X_shape = draw(st.sampled_from([[shape0, shape4, shape5]]))
            Y_shape = draw(st.sampled_from([[shape6, shape5, shape5]]))
        if case == 4:
            x_num_col_dims = 3
            y_num_col_dims = 2
            X_shape = draw(
                st.sampled_from([[shape0, shape3, shape4, shape6],
                                 [shape4, shape6, shape0, shape5, shape4]]))
            Y_shape = draw(st.sampled_from([[shape5, shape4, shape5]]))

        force_fp32_output = draw(st.booleans())

        mul_op = OpConfig(
            type="mul",
            inputs={"X": ["input_data_x"],
                    "Y": ["input_data_y"]},
            outputs={"Out": ["output_data"]},
            attrs={
                "x_num_col_dims": x_num_col_dims,
                "y_num_col_dims": y_num_col_dims,
                "force_fp32_output": force_fp32_output
            })

        program_config = ProgramConfig(
            ops=[mul_op],
            weights={},
            inputs={
                "input_data_x": TensorConfig(shape=X_shape),
                "input_data_y": TensorConfig(shape=Y_shape)
            },
            outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["mul"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        # int8 case run long time in max_examples=250
        self.run_and_statis(quant=False, max_examples=100)


if __name__ == "__main__":
    unittest.main(argv=[''])
