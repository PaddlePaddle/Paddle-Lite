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

from numpy.lib.function_base import place
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


class TestIsEmptyOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        host_op_config = [
            Place(TargetType.Host, PrecisionType.Any, DataLayoutType.Any),
            Place(TargetType.Host, PrecisionType.Any, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=host_op_config)

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=3, max_value=64), min_size=3, max_size=4))
        process_type = draw(
            st.sampled_from(["type_int32", "type_int64", "type_float"]))

        def generate_data(type, size_list):
            if type == "type_int32":
                return np.random.randint(
                    low=0, high=100, size=size_list).astype(np.int32)
            elif type == "type_int64":
                return np.random.randint(
                    low=0, high=100, size=size_list).astype(np.int64)
            elif type == "type_float":
                return np.random.random(size=size_list).astype(np.float32)

        def generate_input_x():
            return generate_data(process_type, in_shape)

        build_ops = OpConfig(
            type="is_empty",
            inputs={"X": ["data_x"]},
            outputs={"Out": ["output_data"], },
            attrs={})
        build_ops.outputs_dtype = {"output_data": np.bool_}

        cast_out = OpConfig(
            type="cast",
            inputs={"X": ["output_data"], },
            outputs={"Out": ["cast_data_out"], },
            attrs={"in_dtype": int(0),
                   "out_dtype": int(2)})
        cast_out.outputs_dtype = {"cast_data_out": np.int32}

        program_config = ProgramConfig(
            ops=[build_ops, cast_out],
            weights={},
            inputs={
                "data_x": TensorConfig(data_gen=partial(generate_input_x))
            },
            outputs=["cast_data_out"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["is_empty"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=60)


if __name__ == "__main__":
    unittest.main(argv=[''])
