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


class TestDepthwiseConv2dOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        # arm_valid_places = [Place(TargetType.ARM, PrecisionType.FP32, DataLayoutType.NCHW),
        #                     Place(TargetType.ARM, PrecisionType.FP16, DataLayoutType.NCHW),
        #                     Place(TargetType.ARM, PrecisionType.INT8, DataLayoutType.NCHW)]
        # self.enable_testing_on_place(places=arm_valid_places, thread=[1,4])
        x86_valid_places = [
            Place(TargetType.X86, PrecisionType.FP32, DataLayoutType.NCHW),
            Place(TargetType.X86, PrecisionType.INT8, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=x86_valid_places, thread=[1, 4])
        # opencl_valid_places = [Place(TargetType.OpenCL, PrecisionType.FP16, DataLayoutType.ImageDefault),
        #                        Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW)]
        # self.enable_testing_on_place(places=opencl_valid_places)

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        # 1. Current not support quantize unit test.
        if program_config.weights['filter_data'].dtype == np.int8:
            return False
        return True

    def sample_program_configs(self, draw):
        input_n = draw(st.integers(min_value=1, max_value=64))
        input_c = draw(st.integers(min_value=1, max_value=64))
        input_h = draw(st.integers(min_value=1, max_value=64))
        input_w = draw(st.integers(min_value=1, max_value=64))
        filter_m = input_c
        filter_c = 1
        filter_h = draw(st.integers(min_value=1, max_value=11))
        filter_w = draw(st.integers(min_value=1, max_value=11))
        scale_in = draw(st.floats(min_value=0.001, max_value=0.1))
        scale_out = draw(st.floats(min_value=0.001, max_value=0.1))
        assume(input_h >= filter_h)
        assume(input_w >= filter_w)
        groups = input_c
        paddings = draw(
            st.lists(
                st.integers(
                    min_value=0, max_value=20), min_size=2, max_size=2))
        dilations = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=2, max_size=2))
        padding_algorithm = draw(
            st.sampled_from(["EXPLICIT", "VALID", "SAME"]))
        strides = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=2, max_size=2))
        data_format = "NCHW"
        use_mkldnn = False

        def calc_output_size():
            if padding_algorithm == "SAME":
                output_h = input_h
                output_w = input_w
            elif padding_algorithm == "VALID":
                output_h = (input_h - (dilations[0] *
                                       (filter_h - 1) + 1)) / strides[0] + 1
                output_w = (input_w - (dilations[1] *
                                       (filter_w - 1) + 1)) / strides[1] + 1
            else:
                output_h = (input_h + 2 * paddings[0] -
                            (dilations[0] *
                             (filter_h - 1) + 1)) / strides[0] + 1
                output_w = (input_w + 2 * paddings[1] -
                            (dilations[1] *
                             (filter_w - 1) + 1)) / strides[1] + 1
            return output_h, output_w

        output_h, output_w = calc_output_size()
        assume(output_h >= 1)
        assume(output_w >= 1)

        in_shape = [input_n, input_c, input_h, input_w]
        weight_shape = [filter_m, filter_c, filter_h, filter_w]

        def gen_data_type():
            input_data_type = draw(st.sampled_from([np.float32, np.int8]))
            filter_data_type = draw(st.sampled_from([np.float32, np.int8]))
            output_data_type = draw(st.sampled_from([np.float32, np.int8]))
            if filter_data_type == np.float32:
                input_data_type = np.float32
                output_data_type == np.float32
            return input_data_type, filter_data_type, output_data_type

        def generate_data(*args, **kwargs):
            if kwargs['dtype'] == np.float32:
                return np.random.random(kwargs['shape']).astype(kwargs[
                    'dtype'])
            elif kwargs['dtype'] == np.int8:
                return np.random.randint(
                    -128, 128, size=kwargs['shape']).astype(kwargs['dtype'])

        def generate_bias(*args, **kwargs):
            if use_mkldnn:
                return np.random.randint(
                    -10, 10, size=kwargs['shape']).astype(kwargs['dtype'])
            else:
                return np.zeros(shape=kwargs['shape']).astype(kwargs['dtype'])

        input_data_type, filter_data_type, output_data_type = gen_data_type()
        depthwise_conv2d_op = OpConfig(
            type="depthwise_conv2d",
            inputs={
                "Input": ["input_data"],
                "Filter": ["filter_data"],
                "Bias": ["bias_data"]
            },
            outputs={"Output": ["output_data"]},
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
        depthwise_conv2d_op.outputs_dtype = {"Output": output_data_type}
        program_config = ProgramConfig(
            ops=[depthwise_conv2d_op],
            weights={
                "filter_data": TensorConfig(data_gen=partial(
                    generate_data, shape=weight_shape,
                    dtype=filter_data_type)),
                "bias_data": TensorConfig(data_gen=partial(
                    generate_bias, shape=[filter_m], dtype=np.float32))
            },
            inputs={
                "input_data": TensorConfig(data_gen=partial(
                    generate_data, shape=in_shape, dtype=input_data_type))
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        config = CxxConfig()
        return self.get_predictor_configs(), ["depthwise_conv2d"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)


if __name__ == "__main__":
    unittest.main(argv=[''])
