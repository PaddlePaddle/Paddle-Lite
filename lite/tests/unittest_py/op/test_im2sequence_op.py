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


class TestIm2sequenceOp(AutoScanTest):
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
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=8, max_value=64), min_size=4, max_size=4))
        stride = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=2), min_size=2, max_size=2))
        out_stride = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=2), min_size=2, max_size=2))
        ker = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=5), min_size=2, max_size=2))
        case_num = draw(st.sampled_from(["c1", "c2"]))
        pad_0 = draw(st.integers(min_value=0, max_value=2))
        pad_1 = draw(st.integers(min_value=0, max_value=2))
        pad = [pad_0, pad_1, pad_0, pad_1]

        def generate_input1(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        def generate_input2(*args, **kwargs):
            a = np.array([in_shape[2], in_shape[3]]).astype(np.float32)
            b = a.repeat(in_shape[0], axis=0)
            return b

        if case_num == "c1":
            build_op = OpConfig(
                type="im2sequence",
                inputs={"X": ["input_data"], },
                outputs={"Out": ["output_data"]},
                attrs={"strides": stride,
                       "paddings": pad,
                       "kernels": ker})

            program_config = ProgramConfig(
                ops=[build_op],
                weights={},
                inputs={"input_data": TensorConfig(data_gen=generate_input1)},
                outputs=["output_data"])
        elif case_num == "c2":
            build_op = OpConfig(
                type="im2sequence",
                inputs={
                    "X": ["input_data"],
                    "Y": ["input_data2"],
                },
                outputs={"Out": ["output_data"]},
                attrs={
                    "strides": stride,
                    "paddings": pad,
                    "kernels": ker,
                    "out_stride": out_stride
                })

            program_config = ProgramConfig(
                ops=[build_op],
                weights={},
                inputs={
                    "input_data": TensorConfig(data_gen=generate_input1),
                    "input_data2": TensorConfig(data_gen=generate_input2)
                },
                outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["im2sequence"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=200)


if __name__ == "__main__":
    unittest.main(argv=[''])
