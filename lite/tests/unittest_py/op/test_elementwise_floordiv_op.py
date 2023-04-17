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

from os import terminal_size
import sys
sys.path.append('../')

from auto_scan_test import AutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st
import numpy as np
from functools import partial
from test_elementwise_add_op import check_broadcast


class TestElementwiseFloorDivOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])
        opencl_valid_places = [
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
        self.enable_testing_on_place(places=opencl_valid_places)

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        target_type = predictor_config.target()
        input_data_type = program_config.inputs["input_data_x"].dtype
        # Check config
        if target_type in [TargetType.ARM]:
            if predictor_config.precision(
            ) == PrecisionType.FP16 and input_data_type != np.float32:
                return False
        return True

    def sample_program_configs(self, draw):
        input_data_x_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=20), min_size=1, max_size=4))
        input_data_y_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=20), min_size=1, max_size=4))
        input_data_x_shape = draw(st.sampled_from([input_data_x_shape, []]))
        input_data_y_shape = draw(st.sampled_from([input_data_y_shape, []]))
        axis = draw(st.integers(min_value=-1, max_value=4))
        assume(
            check_broadcast(input_data_x_shape, input_data_y_shape, axis) ==
            True)
        if axis < 0:
            axis = abs(len(input_data_x_shape) - len(
                input_data_y_shape)) + axis + 1

        if self.get_target().upper() == 'X86':
            input_data_type = draw(
                st.sampled_from([np.float32, np.int32, np.int64]))
        elif self.get_target().upper() == 'ARM':
            input_data_type = draw(st.sampled_from([np.int32, np.int64]))
        elif self.get_target().upper() == 'OPENCL':
            input_data_type = draw(st.sampled_from([np.float32]))
        elif self.get_target().upper() == 'METAL':
            input_data_type = draw(st.sampled_from([np.float32]))

        def gen_input_data(*args, **kwargs):
            return np.random.randint(
                1, 20, size=(kwargs['shape'])).astype(kwargs['dtype'])

        elementwise_floordiv_op = OpConfig(
            type="elementwise_floordiv",
            inputs={"X": ["input_data_x"],
                    "Y": ["input_data_y"]},
            outputs={"Out": ["output_data"]},
            attrs={"axis": axis})
        program_config = ProgramConfig(
            ops=[elementwise_floordiv_op],
            weights={},
            inputs={
                "input_data_x": TensorConfig(data_gen=partial(
                    gen_input_data,
                    shape=input_data_x_shape,
                    dtype=input_data_type)),
                "input_data_y": TensorConfig(data_gen=partial(
                    gen_input_data,
                    shape=input_data_y_shape,
                    dtype=input_data_type))
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["elementwise_floordiv"], (1e-5,
                                                                        1e-5)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            input_data_type = program_config.inputs["input_data_x"].dtype
            if input_data_type == np.float32:
                err_msg = "Paddle's elementwise_floordiv op doesn't support float32 datatype!"
                return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.PADDLE_NOT_SUPPORT,
            "Paddle does not support this op in a specific case. We need to fix it as soon as possible."
        )

        def _teller3(program_config, predictor_config):
            target_type = predictor_config.target()
            in_x_shape = list(program_config.inputs["input_data_x"].shape)
            in_y_shape = list(program_config.inputs["input_data_y"].shape)
            if target_type != TargetType.ARM and target_type != TargetType.X86 and target_type != TargetType.Host:
                if len(in_x_shape) == 0 or len(in_y_shape) == 0:
                    return True

        self.add_ignore_check_case(
            _teller3, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Only test 0D-tensor on CPU(ARM/X86/Host) now.")

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)


if __name__ == "__main__":
    unittest.main(argv=[''])
