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
'''
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
        self.enable_testing_on_place(TargetType.ARM, PrecisionType.FP32,
                                     DataLayoutType.NCHW)

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return False  # ci run on arm_opencl error
        # get input&output shape, get op attributes
        x_shape = list(program_config.inputs["input_data_x"].shape)
        y_shape = list(program_config.weights["input_data_y"].shape)
        x_precision = program_config.inputs["input_data_x"].dtype
        x_num_col_dims = program_config.ops[0].attrs["x_num_col_dims"]
        y_num_col_dims = program_config.ops[0].attrs["y_num_col_dims"]

        # {TargetType.Host, TargetType.X86, TargetType.ARM, TargetType.OpenCL}
        if predictor_config.target() == TargetType.ARM:
            # get input and output shape of current op
            if x_shape[1] != y_shape[0]:
                return False
        # {PrecisionType.FP16, PrecisionType.FP32, PrecisionType.FP64, PrecisionType.UINT8, PrecisionType.INT8, PrecisionType.INT16, PrecisionType.INT32, PrecisionType.INT64, PrecisionType.BOOL}
        target_type = predictor_config.target()
        if target_type not in [TargetType.OpenCL, TargetType.Metal]:
            if predictor_config.precision(
            ) == PrecisionType.FP16 and in_data_type != np.float16:
                return False
            elif predictor_config.precision(
            ) == PrecisionType.FP32 and in_data_type != np.float32:
                return False

        # {DataLayoutType.NCHW, DataLayoutType.NHWC, DataLayoutType.ImageDefault, DataLayoutType.ImageFolder, DataLayoutType.ImageNW, DataLayoutType.Any}
        elif predictor_config.layout() != DataLayoutType.NCHW:
            if y_num_col_dims > 20:
                return False
        return True

    def sample_program_configs(self, draw):
        in_shape1 = draw(
            st.lists(
                st.integers(
                    min_value=20, max_value=200),
                min_size=2,
                max_size=2))
        in_shape2 = draw(
            st.lists(
                st.integers(
                    min_value=20, max_value=200),
                min_size=2,
                max_size=2))
        assume(in_shape1[1] == in_shape2[0])

        mul_op = OpConfig(
            type="mul",
            inputs={"X": ["input_data_x"],
                    "Y": ["input_data_y"]},
            outputs={"Out": ["output_data"]},
            attrs={"x_num_col_dims": 1,
                   "y_num_col_dims": 1})

        program_config = ProgramConfig(
            ops=[mul_op],
            weights={"input_data_y": TensorConfig(shape=in_shape2)},
            inputs={"input_data_x": TensorConfig(shape=in_shape1)},
            outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["mul"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            # get input&output shape, get op attributes
            x_shape = list(program_config.inputs["input_data_x"].shape)
            y_shape = list(program_config.weights["input_data_y"].shape)
            x_num_col_dims = program_config.ops[0].attrs["x_num_col_dims"]
            y_num_col_dims = program_config.ops[0].attrs["y_num_col_dims"]

            # {TargetType.Host, TargetType.X86, TargetType.ARM, TargetType.OpenCL}
            if predictor_config.target() == TargetType.ARM:
                if len(x_shape) > 4:
                    return True
            # {PrecisionType.FP16, PrecisionType.FP32, PrecisionType.FP64, PrecisionType.UINT8, PrecisionType.INT8, PrecisionType.INT16, PrecisionType.INT32, PrecisionType.INT64, PrecisionType.BOOL}
            elif predictor_config.precision() == PrecisionType.FP16:
                if len(y_shape) > 4:
                    return True
            # {DataLayoutType.NCHW, DataLayoutType.NHWC, DataLayoutType.ImageDefault, DataLayoutType.ImageFolder, DataLayoutType.ImageNW, DataLayoutType.Any}
            elif predictor_config.layout() != DataLayoutType.NCHW:
                if x_num_col_dims != y_num_col_dims:
                    return True
            return False

        # ACCURACY_ERROR ignore case will be operated, but we will not check the output precision.
        self.add_ignore_check_case(
            # IgnoreReasonsBase.PADDLE_NOT_IMPLEMENTED
            # IgnoreReasonsBase.PADDLELITE_NOT_SUPPORT
            # IgnoreReasonsBase.ACCURACY_ERROR
            teller1,
            IgnoreReasons.ACCURACY_ERROR,
            "The op output has diff in a specific case. We need to fix it as soon as possible."
        )

        def teller2(program_config, predictor_config):
            if x_num_col_dims != y_num_col_dims:
                return True
            return False

        # PADDLELITE_NOT_SUPPORT ignore case will not be operated.
        self.add_ignore_check_case(
            # IgnoreReasonsBase.PADDLE_NOT_IMPLEMENTED
            # IgnoreReasonsBase.PADDLELITE_NOT_SUPPORT
            # IgnoreReasonsBase.ACCURACY_ERROR
            teller2,
            IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "The format 'x_num_col_dims != y_num_col_dims' is not supported, we need to fix it as soon as possible."
        )

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
'''
