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


class TestTopKV2Op(AutoScanTest):
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
        in_shape = draw(st.sampled_from([[N, C, H, W]]))
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=3, max_value=5), min_size=1, max_size=4))
        in_dtype = draw(st.sampled_from([np.float32]))

        def generate_X_data():
            return np.random.normal(0.0, 5.0, in_shape).astype(in_dtype)

        k_data = draw(st.integers(min_value=1, max_value=2))

        axis_data = draw(st.integers(min_value=0, max_value=1))

        choose_k = draw(st.sampled_from(["k", "K"]))
        inputs = {"X": ["X_data"]}

        def generate_K_data():
            if (choose_k == "K"):
                inputs["K"] = ["K_data"]
                a = np.array([k_data]).astype(np.int32)
                return a
            else:
                return np.random.randint(1, 5, []).astype(np.int32)

        assume(k_data <= in_shape[-1])
        assume(axis_data < len(in_shape))

        # Lite does not have these two attributes
        largest_data = draw(st.booleans())
        sorted_data = draw(st.booleans())

        top_k_v2_op = OpConfig(
            type="top_k_v2",
            inputs=inputs,
            outputs={"Out": ["Out_data"],
                     "Indices": ["Indices_data"]},
            attrs={
                "k": k_data,
                "axis": axis_data,
                #"largest": largest_data,
                #"sorted": sorted_data
            })
        top_k_v2_op.outputs_dtype = {"Out_data": in_dtype}

        program_config = ProgramConfig(
            ops=[top_k_v2_op],
            weights={},
            inputs={
                "X_data": TensorConfig(data_gen=partial(generate_X_data)),
                "K_data": TensorConfig(data_gen=partial(generate_K_data))
            },
            outputs=["Out_data", "Indices_data"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), [""], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
