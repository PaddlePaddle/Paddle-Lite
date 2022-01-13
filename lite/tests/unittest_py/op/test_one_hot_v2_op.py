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


class TestOneHotV2Op(AutoScanTest):
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
        # if max_value > 32 will crash, todo fix
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=32), min_size=2, max_size=2))
        depth_shape = draw(
            st.lists(
                st.integers(
                    min_value=32, max_value=256),
                min_size=1,
                max_size=1))

        # if def depth_tensor  will have crash
        #def generate_depth_tensor(*args, **kwargs):
        #    len = np.ones(1)
        #    len[0] = 8
        #    return len.astype(np.int32)
        def generate_input1(*args, **kwargs):
            return np.random.randint([in_shape]).astype(np.int64)

        dtype = draw(st.sampled_from([2]))
        depth = draw(st.integers(min_value=32, max_value=256))
        allow_out_of_range = draw(st.booleans())
        one_hot_v2_op = OpConfig(
            type="one_hot_v2",
            #inputs = {"X" : ["input_data"], "depth_tensor":["depth_tensor"]},
            inputs={"X": ["input_data"]},
            outputs={"Out": ["output_data"]},
            attrs={
                "depth": depth,
                "dtype": dtype,
                "allow_out_of_range": allow_out_of_range
            })
        program_config = ProgramConfig(
            ops=[one_hot_v2_op],
            weights={},
            inputs={
                "input_data": TensorConfig(
                    shape=in_shape, data_gen=generate_input1)
                #TensorConfig(shape=in_shape, data_gen=generate_input1),
                #"depth_tensor":
                #TensorConfig(shape=depth_shape, data_gen=generate_depth_tensor)
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["one_hot_v2"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            if predictor_config.target() == TargetType.Host:
                return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.ACCURACY_ERROR,
            "The op output has diff in a specific case on host. We need to fix it as soon as possible."
        )

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
