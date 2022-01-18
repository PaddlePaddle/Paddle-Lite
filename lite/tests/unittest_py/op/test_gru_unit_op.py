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


class TestGruUnitOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 2])
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 2, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        shape0 = draw(st.integers(min_value=1, max_value=3))
        shape1 = draw(st.integers(min_value=1, max_value=3))
        shape2 = draw(st.integers(min_value=1, max_value=3))
        lod_arr = [0, shape0, shape0 + shape1, shape0 + shape1 + shape2]

        bool_orimode = draw(st.sampled_from([True, False]))
        shape_0 = draw(st.integers(min_value=1, max_value=60))
        in_shape = [shape_0, shape_0 * 3]
        batch = lod_arr[3]

        def generate_input():
            return np.random.random([batch, in_shape[1]]).astype(np.float32)

        def generate_weight():
            return np.random.random(in_shape).astype(np.float32)

        def generate_hid():
            return np.random.random([batch, in_shape[0]]).astype(np.float32)

        def generate_bias():
            return np.random.random([1, in_shape[1]]).astype(np.float32)

        build_ops = OpConfig(
            type="gru_unit",
            inputs={
                "Input": ["input_data"],
                "HiddenPrev": ["hid_data"],
                "Weight": ["weight_data"],
                "Bias": ["bias_data"]
            },
            outputs={
                "Gate": ["gate"],
                "ResetHiddenPrev": ["reset_hidden"],
                "Hidden": ["hidden"],
            },
            attrs={
                "activation": 2,  #tanh
                "gate_activation": 1,  #sigmoid
                "origin_mode": bool_orimode,
            })
        program_config = ProgramConfig(
            ops=[build_ops],
            weights={},
            inputs={
                "input_data": TensorConfig(
                    data_gen=partial(generate_input), lod=[lod_arr]),
                "hid_data": TensorConfig(data_gen=partial(generate_hid)),
                "weight_data": TensorConfig(data_gen=partial(generate_weight)),
                "bias_data": TensorConfig(data_gen=partial(generate_bias)),
            },
            outputs=["hidden"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["gru_unit"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=100)


if __name__ == "__main__":
    unittest.main(argv=[''])
