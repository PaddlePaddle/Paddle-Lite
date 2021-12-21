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
import argparse
from functools import partial
import random
import numpy as np


class TestConcatOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
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
        target_type = predictor_config.target()
        if target_type == TargetType.OpenCL:
            if "AxisTensor" in program_config.ops[0].inputs:
                return False
        return True

    def sample_program_configs(self, draw):
        in_shape1 = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=100),
                min_size=1,
                max_size=4))
        in_shape2 = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=100),
                min_size=1,
                max_size=4))
        axis = draw(st.sampled_from([-1, 0, 1, 2, 3]))
        has_axis_tensor = draw(st.booleans())
        assume(len(in_shape1) == len(in_shape2))
        assume(axis < len(in_shape1))
        for i in range(-1, len(in_shape1)):
            if i == axis:
                continue
            else:
                assume(in_shape1[i] == in_shape2[i])

        def generate_input1(*args, **kwargs):
            return np.random.random(in_shape1).astype(np.float32)

        def generate_input2(*args, **kwargs):
            return np.random.random(in_shape2).astype(np.float32)

        def generate_axis(*args, **kwargs):
            return np.array([axis]).astype("int32")

        input_type_dict = {"X": ["input_data1", "input_data2"]}
        input_data_dict = {
            "input_data1": TensorConfig(data_gen=partial(generate_input1)),
            "input_data2": TensorConfig(data_gen=partial(generate_input2))
        }
        if has_axis_tensor:
            input_type_dict["AxisTensor"] = ["axis_tensor_data"]
            input_data_dict["axis_tensor_data"] = TensorConfig(
                data_gen=partial(generate_axis))

        concat_op = OpConfig(
            type="concat",
            inputs=input_type_dict,
            outputs={"Out": ["output_data"]},
            attrs={"axis": axis})
        program_config = ProgramConfig(
            ops=[concat_op],
            weights={},
            inputs=input_data_dict,
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["concat"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 25
        if target_str == "OpenCL":
            # Make sure to generate enough valid cases for OpenCL
            max_examples = 100

        self.run_and_statis(
            quant=False, min_success_num=25, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
