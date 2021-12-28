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
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st
import numpy as np
from functools import partial


class TestDeformableConvOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        # self.enable_testing_on_place(TargetType.ARM, PrecisionType.FP32, DataLayoutType.NCHW, thread=[1,4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        input_n = draw(st.integers(min_value=1, max_value=64))
        input_c = draw(st.integers(min_value=1, max_value=128))
        input_h = draw(st.integers(min_value=1, max_value=128))
        input_w = draw(st.integers(min_value=1, max_value=128))
        filter_m = draw(st.integers(min_value=1, max_value=128))
        filter_c = input_c
        filter_h = draw(st.integers(min_value=1, max_value=7))
        filter_w = draw(st.integers(min_value=1, max_value=7))
        assume(input_h >= filter_h)
        assume(input_w >= filter_w)
        groups = draw(st.integers(min_value=1, max_value=input_c))
        assume(groups * filter_c == input_c)
        assume(filter_m % groups == 0)
        deformable_groups = groups  #draw(st.integers(min_value = 1, max_value = filter_m))
        assume(filter_m % deformable_groups == 0)

        paddings = draw(
            st.lists(
                st.integers(
                    min_value=0, max_value=20), min_size=2, max_size=2))
        dilations = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=2, max_size=2))
        strides = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=2, max_size=2))
        im2col_step = 1  #draw(st.sampled_from([32,64]))#draw(st.integers(min_value = 1, max_value = 128))
        assume(input_n % im2col_step == 0)

        in_shape = [input_n, input_c, input_h, input_w]
        weight_shape = [filter_m, filter_c // groups, filter_h, filter_w]
        h_out = (input_h + 2 * paddings[0] -
                 (dilations[0] * (filter_h - 1) + 1)) // strides[0] + 1
        w_out = (input_w + 2 * paddings[1] -
                 (dilations[1] * (filter_w - 1) + 1)) // strides[1] + 1
        assume(h_out >= 1)
        assume(w_out >= 1)
        offset_shape = [
            input_n, deformable_groups * filter_w * filter_h * 2, h_out, w_out
        ]
        mask_shape = [
            input_n, deformable_groups * filter_w * filter_h, h_out, w_out
        ]

        def generate_input(*args, **kwargs):
            return np.random.normal(0.0, 1.0, in_shape).astype(np.float32)

        def generate_filter(*args, **kwargs):
            return np.random.normal(0.0, 1.0, weight_shape).astype(np.float32)

        def generate_offset(*args, **kwargs):
            return np.random.normal(0.0, 1.0, offset_shape).astype(np.float32)

        def generate_mask(*args, **kwargs):
            return np.random.normal(0.0, 1.0, mask_shape).astype(np.float32)

        deformable_conv_op = OpConfig(
            type="deformable_conv",
            inputs={
                "Input": ["input_data"],
                "Filter": ["filter_data"],
                "Offset": ["offset_data"],
                "Mask": ["mask_data"]
            },
            outputs={"Output": ["output_data"]},
            attrs={
                "strides": strides,
                "paddings": paddings,
                "groups": groups,
                "deformable_groups": deformable_groups,
                "dilations": dilations,
                "im2col_step": im2col_step
            })
        program_config = ProgramConfig(
            ops=[deformable_conv_op],
            weights={
                "filter_data": TensorConfig(data_gen=partial(generate_filter)),
                "offset_data": TensorConfig(data_gen=partial(generate_offset)),
                "mask_data": TensorConfig(data_gen=partial(generate_mask)),
            },
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input))
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["deformable_conv"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(
            quant=False,
            max_examples=300,
            reproduce=reproduce_failure('6.31.6', b'AAA5KwE8BgAAAAAAAAAA'))


if __name__ == "__main__":
    unittest.main(argv=[''])
