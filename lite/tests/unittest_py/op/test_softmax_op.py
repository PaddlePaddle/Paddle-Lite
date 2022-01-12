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
from functools import partial
import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st
import numpy as np


class TestSoftmaxOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])
        opencl_places = [
            Place(TargetType.OpenCL, PrecisionType.FP16,
                  DataLayoutType.ImageFolder), Place(
                      TargetType.OpenCL, PrecisionType.FP16,
                      DataLayoutType.ImageDefault),
            Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
            Place(TargetType.OpenCL, PrecisionType.Any,
                  DataLayoutType.ImageDefault), Place(
                      TargetType.OpenCL, PrecisionType.Any,
                      DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(places=opencl_places)
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
        x_shape = list(program_config.inputs["input_data"].shape)
        axis = program_config.ops[0].attrs["axis"]
        if predictor_config.target() == TargetType.OpenCL:
            if len(x_shape) < 2 or (len(x_shape) == 4 and axis == 0):
                return False
        if predictor_config.target() == TargetType.Metal:
            if len(x_shape) <= 3 or x_shape[0] != 1 \
                or (len(x_shape) == 4 and axis == 0):
                return False
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=64), min_size=0, max_size=3))
        in_shape.insert(0, draw(st.integers(min_value=1, max_value=4)))
        input_axis = draw(st.sampled_from([0, 1, 2, 3, -1]))
        assume(input_axis < len(in_shape))

        def generate_input(*args, **kwargs):
            return np.random.normal(0.0, 1.0, in_shape).astype(np.float32)

        ops_config = OpConfig(
            type="softmax",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["output_data"]},
            attrs={"axis": input_axis})

        program_config = ProgramConfig(
            ops=[ops_config],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input))
            },
            outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        target_str = self.get_target()
        if target_str == "Metal":
            atol, rtol = 1e-3, 1e-3
        return self.get_predictor_configs(), ["softmax"], (atol, rtol)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 25
        if target_str == "OpenCL":
            # Make sure to generate enough valid cases for OpenCL
            max_examples = 100
        elif target_str == "Metal":
            # Make sure to generate enough valid cases for Metal
            max_examples = 500
        self.run_and_statis(quant=False, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
