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
import numpy as np
from functools import partial


class TestConv2dOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        x86_places = [
            Place(TargetType.X86, PrecisionType.INT8, DataLayoutType.NCHW),
            Place(TargetType.X86, PrecisionType.FP32, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=x86_places, thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        num = 1
        cin = 1
        cout = 1
        height = draw(st.integers(min_value=1, max_value=8))
        width = draw(st.integers(min_value=1, max_value=8))
        kw = 1
        kh = 1
        groups = 1
        scale_in = draw(st.floats(min_value=0.001, max_value=0.1))
        scale_out = draw(st.floats(min_value=0.001, max_value=0.1))
        assume(cin % groups == 0)
        assume(cout % groups == 0)
        w_cin = (int)(cin / groups)
        in_shape = [num, cin, height, width]
        in_shape = [num, cin, 2, 2]  #remove later
        assume(in_shape[2] == in_shape[3])  #remove later
        weight_shape = [cout, w_cin, kh, kw]
        assume(in_shape[2] >= weight_shape[2])
        assume(in_shape[3] >= weight_shape[3])

        paddings = [0, 0]
        dilations = draw(st.sampled_from([[1, 1]]))
        padding_algorithm = draw(st.sampled_from(["SAME"]))
        strides = draw(st.sampled_from([[1, 1]]))
        data_format = "NCHW"

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        def generate_filter(*args, **kwargs):
            return np.random.random(weight_shape).astype(np.float32)

        def generate_bias(*args, **kwargs):
            return np.random.random([cout]).astype(np.float32)

        def generate_alpha(*args, **kwargs):
            return np.random.random([1]).astype(np.float32)

        conv_op = OpConfig(
            type="conv2d",
            inputs={
                "Input": ["input_data"],
                "Filter": ["filter_data"],
                # "Bias": ["bias_data"]
            },
            outputs={"Output": ["conv_output_data"]},
            attrs={
                "strides": strides,
                "paddings": paddings,
                "use_mkldnn": True,
                "padding_algorithm": padding_algorithm,
                "groups": groups,
                "dilations": dilations,
                "Scale_in": scale_in,
                "Scale_out": scale_out,
                "data_format": data_format
            })

        relu_op = OpConfig(
            type="relu",
            inputs={"X": ["conv_output_data"]},
            outputs={"Out": ["output_data"]},
            attrs={"data_format": "NCHW"})

        ops = [conv_op, relu_op]
        program_config = ProgramConfig(
            ops=ops,
            weights={
                "filter_data": TensorConfig(data_gen=partial(generate_filter)),
                # "bias_data": TensorConfig(data_gen=partial(generate_bias))
            },
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input)),
                # "alpha_data": TensorConfig(data_gen=partial(generate_alpha))
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        config = CxxConfig()
        return self.get_predictor_configs(), ["conv2d", "relu"], (1e-3, 1e-3)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=True, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
