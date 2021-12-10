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
        opencl_places = [Place(TargetType.OpenCL, PrecisionType.FP16, DataLayoutType.ImageDefault),
                          Place(TargetType.OpenCL, PrecisionType.FP16, DataLayoutType.ImageFolder),
                          Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
                          Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.ImageDefault),
                          Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.ImageFolder),
                          Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
                          Place(TargetType.Host, PrecisionType.FP32)]
        self.enable_testing_on_place(places=opencl_places)

    def is_program_valid(self, program_config: ProgramConfig , predictor_config: CxxConfig) -> bool:
        if predictor_config.target() == TargetType.ARM:
            if predictor_config.precision() == PrecisionType.FP16 or predictor_config.precision() == PrecisionType.INT8:
                return False
            else:
                return True
        else:
            return True

    def sample_program_configs(self, draw):
        num = draw(st.integers(min_value=1, max_value=64))
        w_cin = draw(st.integers(min_value=1, max_value=128))
        height = draw(st.integers(min_value=1, max_value=128))
        width = draw(st.integers(min_value=1, max_value=128))
        cout = draw(st.integers(min_value=1, max_value=128))
        kw = np.random.randint(1, 9)
        kh = np.random.randint(1, 9)
        groups = draw(st.sampled_from([1, 2, 128]))
        scale_in = draw(st.floats(min_value=0.001, max_value=0.1))
        scale_out = draw(st.floats(min_value=0.001, max_value=0.1))
        cin = w_cin * groups
        in_shape = [num, cin, height, width]
        weight_shape = [cout, w_cin, kh, kw]
        assume(in_shape[2] >= weight_shape[2])
        assume(in_shape[3] >= weight_shape[3])

        paddings = draw(st.lists(st.integers(min_value=0, max_value=2), min_size=2, max_size=2))
        dilations = draw(st.sampled_from([[1, 1]]))
        padding_algorithm = draw(st.sampled_from(["VALID", "SAME"]))
        strides = draw(st.sampled_from([[1, 1], [2, 2]]))
        data_format = "NCHW"
        use_mkldnn = False
        if self.target[0] == "X86":
            use_mkldnn = True

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)
        def generate_filter(*args, **kwargs):
            return np.random.random(weight_shape).astype(np.float32)
        def generate_bias(*args, **kwargs):
            return np.random.random([cout]).astype(np.float32)
        inputs_data = {"input_data":
                TensorConfig(data_gen=partial(generate_input))}
        inputs_type = {"Input": ["input_data"], "Filter" : ["filter_data"]}
        if use_mkldnn:
            inputs_data["bias_data"] = TensorConfig(data_gen=partial(generate_bias))
            inputs_type["Bias"] = ["bias_data"]
        
        conv_op = OpConfig(
            type = "conv2d",
            inputs = inputs_type,
            outputs = {"Output": ["output_data"]},
            attrs = {"strides" : strides,
                    "paddings" : paddings,
                    "use_mkldnn" : use_mkldnn,
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
                TensorConfig(data_gen=partial(generate_filter))
            },
            inputs=inputs_data,
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
