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


class TestIndexSelectOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.Any,
            DataLayoutType.NCHW,
            thread=[1, 2])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=6, max_value=64), min_size=4, max_size=4))
        idx_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=5), min_size=1, max_size=1))
        dim_data = draw(st.sampled_from([1, 2, 3]))
        type_str = draw(
            st.sampled_from(
                ["type_float", "type_int", "type_short", "type_char"]))

        def generate_input1(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        def generate_input2(*args, **kwargs):
            return np.random.randint(
                low=1, high=100, size=in_shape).astype(np.int32)

        def generate_input3(*args, **kwargs):
            return np.random.randint(
                low=1, high=100, size=in_shape).astype(np.int16)

        def generate_input4(*args, **kwargs):
            return np.random.randint(
                low=1, high=100, size=in_shape).astype(np.int8)

        def generate_idx(*args, **kwargs):
            return np.random.randint(
                low=1, high=5, size=idx_shape).astype(np.int64)

        build_ops = OpConfig(
            type="index_select",
            inputs={"X": ["input_data"],
                    "Index": ["idx"]},
            outputs={"Out": ["output_data"], },
            attrs={"dim": dim_data, })
        tmp_input = {}
        if type_str == "type_float":
            tmp_input = {
                "input_data": TensorConfig(data_gen=partial(generate_input1)),
                "idx": TensorConfig(data_gen=partial(generate_idx)),
            }
        elif type_str == "type_int":
            build_ops.outputs_dtype = {"output_data": np.int32}
            tmp_input = {
                "input_data": TensorConfig(data_gen=partial(generate_input2)),
                "idx": TensorConfig(data_gen=partial(generate_idx)),
            }
        elif type_str == "type_short":
            build_ops.outputs_dtype = {"output_data": np.int16}
            tmp_input = {
                "input_data": TensorConfig(data_gen=partial(generate_input3)),
                "idx": TensorConfig(data_gen=partial(generate_idx)),
            }
        elif type_str == "type_char":
            build_ops.outputs_dtype = {"output_data": np.int8}
            tmp_input = {
                "input_data": TensorConfig(data_gen=partial(generate_input4)),
                "idx": TensorConfig(data_gen=partial(generate_idx)),
            }
        program_config = ProgramConfig(
            ops=[build_ops],
            weights={},
            inputs=tmp_input,
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["index_select"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=50)


if __name__ == "__main__":
    unittest.main(argv=[''])
