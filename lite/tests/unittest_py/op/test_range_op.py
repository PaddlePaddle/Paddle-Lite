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


class TestRangeOp(AutoScanTest):
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
        input_start = draw(st.sampled_from([0, 1, 2, 10, 20]))
        input_end = draw(st.sampled_from([1, 5, 10, 50, 100]))
        input_step = draw(st.sampled_from([0.1, 0.2, 0.5, 2, 5]))
        input_type = draw(
            st.sampled_from(["type_float", "type_int", "type_int64"]))

        assume(input_start < input_end)
        if input_type != "type_float":
            assume(input_start >= 10)
            assume(input_step >= 1)

        def generate_start(*args, **kwargs):
            if input_type == "type_float":
                return np.array([input_start]).astype(np.float32)
            elif input_type == "type_int":
                return np.array([input_start]).astype(np.int32)
            elif input_type == "type_int64":
                return np.array([input_start]).astype(np.int64)

        def generate_end(*args, **kwargs):
            if input_type == "type_float":
                return np.array([input_end]).astype(np.float32)
            elif input_type == "type_int":
                return np.array([input_end]).astype(np.int32)
            elif input_type == "type_int64":
                return np.array([input_end]).astype(np.int64)

        def generate_step(*args, **kwargs):
            if input_type == "type_float":
                return np.array([input_step]).astype(np.float32)
            elif input_type == "type_int":
                return np.array([input_step]).astype(np.int32)
            elif input_type == "type_int64":
                return np.array([input_step]).astype(np.int64)

        ops_config = OpConfig(
            type="range",
            inputs={"Start": ["start"],
                    "End": ["end"],
                    "Step": ["step"]},
            outputs={"Out": ["output_data"]},
            attrs={}, )

        program_config = ProgramConfig(
            ops=[ops_config],
            weights={},
            inputs={
                "start": TensorConfig(data_gen=partial(generate_start)),
                "end": TensorConfig(data_gen=partial(generate_end)),
                "step": TensorConfig(data_gen=partial(generate_step)),
            },
            outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["range"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
