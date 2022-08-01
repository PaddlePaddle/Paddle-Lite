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


class TestReduceAllOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
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
                    min_value=1, max_value=64), min_size=4, max_size=4))
        keep_dim = draw(st.booleans())
        axis = draw(st.integers(min_value=-1, max_value=3))
        assume(axis < len(in_shape))
        if isinstance(axis, int):
            axis = [axis]
        reduce_all_data = True if axis == None or axis == [] else False

        def generate_input(*args, **kwargs):
            return np.random.randint(
                low=0, high=1, size=in_shape).astype(np.int32)

        cast_x = OpConfig(
            type="cast",
            inputs={"X": ["input_data_x"], },
            outputs={"Out": ["cast_data_x"], },
            attrs={"in_dtype": int(2),
                   "out_dtype": int(0)})
        cast_x.outputs_dtype = {"cast_data_x": np.bool_}

        build_ops = OpConfig(
            type="reduce_all",
            inputs={"X": ["cast_data_x"], },
            outputs={"Out": ["output_data"], },
            attrs={
                "dim": axis,
                "keep_dim": keep_dim,
                "reduce_all": reduce_all_data,
            })

        cast_out = OpConfig(
            type="cast",
            inputs={"X": ["output_data"], },
            outputs={"Out": ["cast_data_out"], },
            attrs={"in_dtype": int(0),
                   "out_dtype": int(2)})
        cast_out.outputs_dtype = {"cast_data_out": np.int32}

        program_config = ProgramConfig(
            ops=[cast_x, build_ops, cast_out],
            weights={},
            inputs={
                "input_data_x": TensorConfig(data_gen=partial(generate_input)),
            },
            outputs=["cast_data_out"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["reduce_all"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=100)


if __name__ == "__main__":
    unittest.main(argv=[''])
