# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from functools import partial
import random
import numpy as np


class TestBmmOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)

        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(
            device_names=["nvidia_tensorrt", "intel_openvino"])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):

        N = draw(st.integers(min_value=2, max_value=4))
        M = draw(st.integers(min_value=2, max_value=64))
        K = draw(st.integers(min_value=2, max_value=64))
        P = draw(st.integers(min_value=2, max_value=64))
        X_shape = draw(st.sampled_from([[N, M, K]]))
        Y_shape = draw(st.sampled_from([[N, K, P]]))
        in_dtype = draw(st.sampled_from([np.float32, ]))

        def generate_X_data():
            return np.random.normal(0.0, 5.0, X_shape).astype(in_dtype)

        def generate_Y_data():
            return np.random.normal(0.0, 5.0, Y_shape).astype(in_dtype)

        bmm_op = OpConfig(
            type="bmm",
            inputs={"X": ["input_data_x"],
                    "Y": ["input_data_y"]},
            outputs={"Out": ["output_data"]}, )
        program_config = ProgramConfig(
            ops=[bmm_op],
            weights={},
            inputs={
                "input_data_x":
                TensorConfig(data_gen=partial(generate_X_data)),
                "input_data_y": TensorConfig(data_gen=partial(generate_Y_data))
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["bmm"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=200)


if __name__ == "__main__":
    unittest.main(argv=[''])
