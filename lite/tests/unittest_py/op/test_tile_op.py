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


class TestTileOp(AutoScanTest):
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

        N = draw(st.integers(min_value=1, max_value=4))
        C = draw(st.integers(min_value=1, max_value=128))
        H = draw(st.integers(min_value=1, max_value=128))
        W = draw(st.integers(min_value=1, max_value=128))
        in_shape = draw(st.sampled_from([[N, C, H, W], [N, H, W]]))

        in_dtype = draw(st.sampled_from([np.float32, np.int32, np.int64]))

        def generate_X_data():
            return np.random.normal(0.0, 5.0, in_shape).astype(in_dtype)

        repeat_times_data = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=5), min_size=1, max_size=4))

        choose_repeat = draw(
            st.sampled_from(
                ["RepeatTimes", "repeat_times_tensor", "repeat_times"]))

        inputs = {"X": ["X_data"]}

        def generate_RepeatTimes_data():
            if (choose_repeat == "RepeatTimes"):
                inputs["RepeatTimes"] = ["RepeatTimes_data"]
                return np.array(repeat_times_data).astype(np.int32)
            else:
                return np.random.randint(1, 5, []).astype(np.int32)

        repeat_times_len = len(repeat_times_data)

        # repeat_times_tensor having repeat_times_len inputs!
        def generate_repeat_times_tensor_data(i):
            inputs["repeat_times_tensor"] = [
                "repeat_times_tensor_data" + str(i)
                for i in range(repeat_times_len)
            ]
            if (choose_repeat == "repeat_times_tensor"):
                return np.array([repeat_times_data[i]]).astype(np.int32)
            else:
                return np.random.randint(1, 5, []).astype(np.int32)

        tile_op = OpConfig(
            type="tile",
            inputs=inputs,
            outputs={"Out": ["Out_data"]},
            attrs={"repeat_times": repeat_times_data})

        program_input = {
            "X_data": TensorConfig(data_gen=partial(generate_X_data)),
            "RepeatTimes_data":
            TensorConfig(data_gen=partial(generate_RepeatTimes_data))
        }
        for i in range(repeat_times_len):
            program_input["repeat_times_tensor_data" + str(i)] = TensorConfig(
                data_gen=partial(generate_repeat_times_tensor_data, i))
        program_config = ProgramConfig(
            ops=[tile_op],
            weights={},
            inputs=program_input,
            outputs=["Out_data"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), [""], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
