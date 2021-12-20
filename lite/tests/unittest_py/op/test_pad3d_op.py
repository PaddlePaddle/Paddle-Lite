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


class TestPad3dOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 2])
        self.enable_testing_on_place(
            TargetType.X86,
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
                    min_value=1, max_value=10), min_size=4, max_size=4))

        mode = draw(
            st.sampled_from(["constant", "reflect", "replicate", "circular"]))
        data_format = draw(st.sampled_from(['NCDHW', 'NDHWC']))
        value_data = draw(st.floats(min_value=0.0, max_value=4.0))
        padding_data = draw(st.sampled_from([[2, 2, 1, 1, 0, 0]]))

        in_shape.insert(0, 1)

        # assume(in_shape in [3, 4, 5])

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        build_ops = OpConfig(
            type="pad3d",
            inputs={
                "X": ["input_data"],
                #"Paddings": padding_data,
            },
            outputs={"Out": ["output_data"], },
            attrs={
                "paddings": padding_data,
                "mode": mode,
                "value": value_data,
                # "data_format": "NCDHW",
                "data_format": data_format,
            })
        program_config = ProgramConfig(
            ops=[build_ops],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input)),
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["pad3d"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
