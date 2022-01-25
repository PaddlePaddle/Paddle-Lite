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


class TestUnfoldOp(AutoScanTest):
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
        C = draw(st.integers(min_value=2, max_value=64))
        H = draw(st.integers(min_value=2, max_value=64))
        W = draw(st.integers(min_value=2, max_value=64))
        in_shape = draw(st.sampled_from([[N, C, H, W]]))
        in_dtype = draw(st.sampled_from([np.float32, ]))

        def generate_X_data():
            return np.random.normal(0.0, 5.0, in_shape).astype(in_dtype)

        paddings = draw(
            st.lists(
                st.integers(
                    min_value=0, max_value=10), min_size=4, max_size=4))
        dilations = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=2, max_size=2))
        strides = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=2, max_size=2))
        kernel_sizes = draw(st.sampled_from([[1, 1], [2, 2], [3, 3]]))

        assume(in_shape[2] >= kernel_sizes[0])
        assume(in_shape[3] >= kernel_sizes[1])

        def cal_output_size(input_size, filter_size, dilation, padding1,
                            padding2, stride):
            dkernel = dilation * (filter_size - 1) + 1
            return (input_size + padding1 + padding2 - dkernel) // stride + 1

        output_height = cal_output_size(in_shape[2], kernel_sizes[0],
                                        dilations[0], paddings[0], paddings[2],
                                        strides[0])
        output_width = cal_output_size(in_shape[3], kernel_sizes[1],
                                       dilations[1], paddings[1], paddings[3],
                                       strides[1])
        assume(output_height > 0)
        assume(output_width > 0)

        unfold_op = OpConfig(
            type="unfold",
            inputs={"X": ["input_data"]},
            outputs={"Y": ["output_data"]},
            attrs={
                "kernel_sizes": kernel_sizes,
                "strides": strides,
                "dilations": dilations,
                "paddings": paddings
            })
        program_config = ProgramConfig(
            ops=[unfold_op],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_X_data)),
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["unfold"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=200)


if __name__ == "__main__":
    unittest.main(argv=[''])
