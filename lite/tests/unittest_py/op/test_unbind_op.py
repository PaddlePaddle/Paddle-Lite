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
from functools import partial
import random
import numpy as np


class TestUnbindOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        host_places = [
            Place(TargetType.Host, PrecisionType.FP32, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=host_places, thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        N = draw(st.integers(min_value=2, max_value=4))
        C = draw(st.integers(min_value=2, max_value=128))
        H = draw(st.integers(min_value=2, max_value=128))
        W = draw(st.integers(min_value=2, max_value=128))
        in_shape = draw(st.sampled_from([[N, C, H, W]]))

        # host only register float32 and int64
        in_dtype = draw(st.sampled_from([np.float32, np.int64]))

        def generate_X_data():
            return np.random.normal(0.0, 5.0, in_shape).astype(in_dtype)

        axis_data = draw(st.integers(min_value=1, max_value=3))

        output_string = ["out"] * in_shape[axis_data]
        for i in range(in_shape[axis_data]):
            output_string[i] += str(i)

        unbind_op = OpConfig(
            type="unbind",
            inputs={"X": ["input_data"]},
            outputs={"Out": output_string},
            attrs={"axis": axis_data})
        program_config = ProgramConfig(
            ops=[unbind_op],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_X_data)),
            },
            outputs=output_string)
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), [""], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
