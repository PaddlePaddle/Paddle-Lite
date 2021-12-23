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
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st
import numpy as np
from functools import partial


class TestElementwiseAddOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM, [
                PrecisionType.FP32, PrecisionType.FP16, PrecisionType.INT32,
                PrecisionType.INT64
            ],
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
        in_x_shape = list(program_config.inputs["input_data_x"].shape)
        in_y_shape = list(program_config.inputs["input_data_y"].shape)
        in_data_type = program_config.inputs["input_data_x"].dtype
        if target_type == TargetType.Metal:
            if in_data_type != np.float32:
                return False
            if in_x_shape != in_y_shape:
                return False
            if in_x_shape[0] != 1:
                return False

        return True

    def sample_program_configs(self, draw):
        input_data_x_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=64), min_size=3, max_size=3))
        input_data_x_shape.insert(
            0, draw(st.integers(
                min_value=1, max_value=3)))
        axis = draw(
            st.integers(
                min_value=0, max_value=(len(input_data_x_shape) - 1)))
        input_data_y_shape = input_data_x_shape[axis:]
        input_data_type = draw(st.sampled_from([np.float32]))

        def gen_input_data(*args, **kwargs):
            return np.random.randint(
                1, 20, size=(kwargs['shape'])).astype(kwargs['dtype'])

        elementwise_add_op = OpConfig(
            type="elementwise_add",
            inputs={"X": ["input_data_x"],
                    "Y": ["input_data_y"]},
            outputs={"Out": ["output_data"]},
            attrs={"axis": axis})
        program_config = ProgramConfig(
            ops=[elementwise_add_op],
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
        config = CxxConfig()
        return self.get_predictor_configs(), ["elementwise_add"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        err_msg = ""

        def teller1(program_config, predictor_config):
            target_type = predictor_config.target()
            in_data_type = program_config.inputs["input_data_x"].dtype
            if target_type in [TargetType.OpenCL, TargetType.Metal]:
                if in_data_type != np.float32:
                    err_msg = "Elementwise_add op on" + str(
                        target_type
                    ) + "backend only support input data type[float32/float16], but got" + str(
                        in_data_type)
                    return True
            return False

        self.add_ignore_check_case(
            teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT, err_msg)

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)


if __name__ == "__main__":
    unittest.main(argv=[''])
