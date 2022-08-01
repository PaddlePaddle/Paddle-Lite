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


class TestOneHotOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        # 128 cost much time, so change to 64
        in_shape0 = draw(st.integers(min_value=1, max_value=64))
        in_shape1 = draw(st.integers(min_value=1, max_value=64))
        in_shape2 = draw(st.integers(min_value=1, max_value=64))

        def generate_depth_tensor(*args, **kwargs):
            return np.random.randint(128, 256, size=1).astype(np.int32)

        in_shape = draw(
            st.sampled_from([(in_shape0, in_shape1, in_shape2, 1), (
                in_shape0, in_shape1, 1), (in_shape0, 1)]))

        def generate_input1(*args, **kwargs):
            return np.random.randint(0, 128, size=in_shape).astype(np.int64)

        dtype = draw(st.sampled_from([2]))
        depth = draw(st.integers(min_value=128, max_value=256))
        allow_out_of_range = draw(st.booleans())
        test_case = draw(st.booleans())
        if test_case:
            one_hot_op = OpConfig(
                type="one_hot",
                inputs={
                    "X": ["input_data"],
                    "depth_tensor": ["depth_tensor"]
                },
                outputs={"Out": ["output_data"]},
                attrs={
                    "depth": depth,
                    "dtype": dtype,
                    "allow_out_of_range": allow_out_of_range
                })
            program_config = ProgramConfig(
                ops=[one_hot_op],
                weights={},
                inputs={
                    "input_data": TensorConfig(
                        shape=list(in_shape), data_gen=generate_input1),
                    "depth_tensor": TensorConfig(
                        shape=[1], data_gen=generate_depth_tensor)
                },
                outputs=["output_data"])
        else:
            one_hot_op = OpConfig(
                type="one_hot",
                inputs={"X": ["input_data"]},
                outputs={"Out": ["output_data"]},
                attrs={
                    "depth": depth,
                    "dtype": dtype,
                    "allow_out_of_range": allow_out_of_range
                })
            program_config = ProgramConfig(
                ops=[one_hot_op],
                weights={},
                inputs={
                    "input_data": TensorConfig(
                        shape=list(in_shape), data_gen=generate_input1)
                },
                outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["one_hot"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
