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


class TestSequenceUnpadOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        length = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=5), min_size=4, max_size=10))
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=5, max_value=10), min_size=1, max_size=3))
        in_shape.insert(0, len(length))
        input_type = draw(st.sampled_from(["float32", "int64"]))

        def generate_input(*args, **kwargs):
            if input_type == "float32":
                return np.random.normal(1.0, 6.0, in_shape).astype(np.float32)
            elif input_type == "int64":
                return np.random.normal(1.0, 6.0, in_shape).astype(np.int64)

        def generate_length(*args, **kwargs):
            return np.array(length).astype('int64')

        ops_config = OpConfig(
            type="sequence_unpad",
            inputs={
                "X": ["input_data"],
                "Length": ["length_data"],
            },
            outputs={"Out": ["output_data"]},
            attrs={})

        program_config = ProgramConfig(
            ops=[ops_config],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input)),
                "length_data": TensorConfig(data_gen=partial(generate_length))
            },
            outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["sequence_unpad"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
