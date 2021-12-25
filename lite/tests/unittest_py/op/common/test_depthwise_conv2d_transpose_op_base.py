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
sys.path.append('..')

from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st


def sample_program_configs(draw):
    input_n = draw(st.integers(min_value=1, max_value=16))
    input_c = draw(st.integers(min_value=1, max_value=16))
    input_h = draw(st.integers(min_value=1, max_value=16))
    input_w = draw(st.integers(min_value=1, max_value=16))
    filter_m = input_c
    filter_c = 1
    filter_h = draw(st.integers(min_value=1, max_value=16))
    filter_w = draw(st.integers(min_value=1, max_value=16))
    dilations = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=16), min_size=2, max_size=2))
    strides = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=16), min_size=2, max_size=2))
    paddings = draw(
        st.lists(
            st.integers(
                min_value=0, max_value=16), min_size=4, max_size=4))
    output_padding = draw(
        st.sampled_from([[], draw(
            st.lists(
                st.integers(
                    min_value=0, max_value=16), min_size=2, max_size=2))])
    )  # draw(st.lists(st.integers(min_value = 0, max_value = 0), min_size = 2, max_size = 2))
    output_size = [
    ]  # draw(st.lists(st.integers(min_value = 0, max_value = 1), min_size = 0, max_size=2))
    groups = input_c
    data_format = draw(st.sampled_from(['NCHW']))
    padding_algorithm = draw(st.sampled_from(['EXPLICIT', 'VALID', 'SAME']))
    use_mkldnn = False

    def generate_input():
        input_shape = []
        if data_format == 'NCHW':
            input_shape = [input_n, input_c, input_h, input_w]
        elif data_format == 'NHWC':
            input_shape = [input_n, input_h, input_w, input_c]
        return np.random.random(input_shape).astype(np.float32)

    def generate_filter():
        filter_shape = [filter_c, filter_m, filter_h,
                        filter_w]  # data_format = 'CMHW'
        return np.random.random(filter_shape).astype(np.float32)

    def generate_bias():
        bias_shape = [filter_m, 1]
        return np.random.random(bias_shape).astype(np.float32)

    def generate_op_inputs():
        inputs = {"Input": ["input"], "Filter": ["filter"]}
        inputs_tensor = {
            "input": TensorConfig(data_gen=partial(generate_input))
        }
        weights_tensor = {
            'filter': TensorConfig(data_gen=partial(generate_filter))
        }
        if draw(st.booleans()) and use_mkldnn:
            inputs["Bias"] = ["bias"]
            weights_tensor["bias"] = TensorConfig(
                data_gen=partial(generate_bias))
        return inputs, inputs_tensor, weights_tensor

    inputs, inputs_tensor, weights_tensor = generate_op_inputs()
    depthwise_conv2d_transpose_op = OpConfig(
        type="depthwise_conv2d_transpose",
        inputs=inputs,
        outputs={"Output": ["output"]},
        attrs={
            "output_padding": output_padding,
            "output_size": output_size,
            "groups": groups,
            "dilations": dilations,
            "strides": strides,
            "paddings": paddings,
            "data_format": data_format,
            "padding_algorithm": padding_algorithm
        })

    program_config = ProgramConfig(
        ops=[depthwise_conv2d_transpose_op],
        weights=weights_tensor,
        inputs=inputs_tensor,
        outputs=["output"])

    return program_config
