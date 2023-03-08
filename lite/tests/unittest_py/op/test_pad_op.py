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

import numpy as np
from functools import partial


class TestPadOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.Any,
            thread=[1, 2])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        # input_dim's size belong to {1~6}, and 2 * input_dim.size() == paddings.size()
        in_shape_case1 = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=8), min_size=1, max_size=6))
        padding_data = draw(
            st.lists(
                st.integers(
                    min_value=0, max_value=3),
                min_size=2 * len(in_shape_case1),
                max_size=2 * len(in_shape_case1)))
        value_data = draw(st.floats(min_value=0.0, max_value=4.0))
        input_data_type = draw(
            st.sampled_from([np.float32, np.int32, np.int64]))

        def generate_input(*args, **kwargs):
            return np.random.randint(
                1, 20, size=in_shape_case1).astype(input_data_type)

        build_ops = OpConfig(
            type="pad",
            inputs={"X": ["input_data"], },
            outputs={"Out": ["output_data"], },
            attrs={"paddings": padding_data,
                   "pad_value": value_data})
        program_config = ProgramConfig(
            ops=[build_ops],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(
                    generate_input, dtype=input_data_type))
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        return self.get_predictor_configs(), ["pad"], (atol, rtol)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=100)


if __name__ == "__main__":
    unittest.main(argv=[''])
