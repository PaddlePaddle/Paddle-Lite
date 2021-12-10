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
        self.enable_testing_on_place(TargetType.X86, PrecisionType.FP32, DataLayoutType.NCHW, thread=[1,4])
        self.enable_testing_on_place(TargetType.ARM, [PrecisionType.FP32,PrecisionType.FP16,PrecisionType.INT8], DataLayoutType.NCHW, thread=[1,4])

    def is_program_valid(self, program_config: ProgramConfig , predictor_config: CxxConfig) -> bool:
        if predictor_config.target() == TargetType.ARM:
            return False
        else:
            return True

    def sample_program_configs(self, draw):
        in_shape=draw(st.lists(st.integers(min_value=1, max_value=64), min_size=4, max_size=4))
        kw = np.random.randint(1, 9)
        kh = np.random.randint(1, 9)
        cout = np.random.randint(1, 128)
        cin = np.random.randint(1, 128)
        scale_in = draw(st.floats(min_value=0.001, max_value=0.1))
        scale_out = draw(st.floats(min_value=0.001, max_value=0.1))
        weight_shape = [cout, cin, kh, kw]
        groups = draw(st.sampled_from([1, 2, cin]))
        val = in_shape[1] * groups
        assume(val == cin)
        assume(in_shape[1] == weight_shape[1])
        assume(in_shape[2] >= weight_shape[2])
        assume(in_shape[3] >= weight_shape[3])

        paddings = draw(st.lists(st.integers(min_value=0, max_value=2), min_size=2, max_size=2))
        dilations = draw(st.sampled_from([[1, 1]]))
        padding_algorithm = draw(st.sampled_from(["VALID", "SAME"]))
        strides = draw(st.sampled_from([[1, 1], [2, 2]]))
        data_format = "NCHW"

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)
        def generate_filter(*args, **kwargs):
            return np.random.random(weight_shape).astype(np.float32)
        def generate_bias(*args, **kwargs):
            return np.random.random([cout]).astype(np.float32)
        conv_op = OpConfig(
            type = "conv2d",
            inputs = {"Input" : ["input_data"],
                     "Filter" : ["filter_data"],
                     "Bias" : ["bias_data"]},
            outputs = {"Output": ["output_data"]},
            attrs = {"strides" : strides,
                    "paddings" : paddings,
                    "use_mkldnn" : True,
                    "padding_algorithm" : padding_algorithm,
                    "groups" : groups,
                    "dilations" : dilations,
                    "Scale_in" : scale_in,
                    "Scale_out" : scale_out,
                    "data_format" : data_format})
        program_config = ProgramConfig(
            ops=[conv_op],
            weights={
                "filter_data":
                TensorConfig(data_gen=partial(generate_filter)),
                "bias_data":
                TensorConfig(data_gen=partial(generate_bias))
            },
            inputs={
                "input_data":
                TensorConfig(data_gen=partial(generate_input))
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        config = CxxConfig()
        return self.get_predictor_configs(), ["conv2d"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)

if __name__ == "__main__":
    unittest.main(argv=[''])
