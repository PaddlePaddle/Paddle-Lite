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
sys.path.append('.')

from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
from test_conv_util import UpdatePaddingAndDilation, ConvOutputSize
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


def sample_program_configs(draw):
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=64), min_size=4, max_size=4))
    weight_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=8), min_size=4, max_size=4))
    paddings = draw(
        st.sampled_from([[1, 2], [4, 2], [1, 1], [0, 0], [1, 0], [1, 1]]))
    dilations = draw(st.sampled_from([[1, 1], [2, 2]]))
    groups = draw(st.sampled_from([1, 2, in_shape[1]]))
    padding_algorithm = draw(st.sampled_from(["VALID", "SAME"]))
    strides = draw(st.sampled_from([[1, 1], [2, 2]]))
    threshold = draw(st.floats(min_value=0, max_value=1))
    alpha = draw(st.floats(min_value=0, max_value=1))
    scale = draw(st.floats(min_value=0.5, max_value=5))
    offset = draw(st.floats(min_value=0, max_value=1))

    assume(in_shape[1] == weight_shape[1] * groups)
    assume(weight_shape[0] % groups == 0)

    paddings_, dilations_ = UpdatePaddingAndDilation(
        in_shape, weight_shape, paddings, dilations, groups, padding_algorithm,
        strides)
    conv_out_shape = [in_shape[0], weight_shape[0]]
    oh, ow = ConvOutputSize(in_shape, weight_shape, dilations_, paddings_,
                            strides)
    conv_out_shape = conv_out_shape + [oh, ow]
    assume(oh > 0 and ow > 0)

    Alpha_shape = []
    mode_data = draw(st.sampled_from(["all", "channel", "element"]))
    if mode_data == "all":
        Alpha_shape = [1]
    elif mode_data == "channel":
        Alpha_shape = [conv_out_shape[1]]
    elif mode_data == "element":
        Alpha_shape = conv_out_shape

    act_type = draw(
        st.sampled_from(
            ['relu', 'relu6', 'leaky_relu', 'hard_swish', 'prelu']))

    def generate_act_attrs(act_type_str):
        attrs = {}
        if act_type_str == 'relu6':
            attrs = {"threshold": threshold}
        if act_type_str == 'leaky_relu':
            attrs = {"alpha": alpha}
        if act_type_str == 'hard_swish':
            attrs = {"threshold": threshold, "scale": scale, "offset": offset}
        if act_type_str == "prelu":
            attrs = {"mode": mode_data, "data_format": "NCHW"}
        return attrs

    conv_op = OpConfig(
        type="conv2d",
        inputs={"Input": ["input_data"],
                "Filter": ["weight_data"]},
        outputs={"Output": ["conv_output_data"]},
        attrs={
            "data_format": 'nchw',
            "dilations": dilations,
            "padding_algorithm": padding_algorithm,
            "groups": groups,
            "paddings": paddings,
            "strides": strides
        })
    active_op_input = {}
    inputs_data = {}
    if act_type == "prelu":
        active_op_input = {"X": ["conv_output_data"], "Alpha": ["alpha_data"]}
        inputs_data = {
            "input_data": TensorConfig(shape=in_shape),
            "alpha_data": TensorConfig(shape=Alpha_shape)
        }
    else:
        active_op_input = {"X": ["conv_output_data"]}
        inputs_data = {"input_data": TensorConfig(shape=in_shape)}

    active_op = OpConfig(
        type=act_type,
        inputs=active_op_input,
        outputs={"Out": ["output_data"]},
        attrs=generate_act_attrs(act_type))

    ops = [conv_op, active_op]
    program_config = ProgramConfig(
        ops=ops,
        weights={"weight_data": TensorConfig(shape=weight_shape)},
        inputs=inputs_data,
        outputs=["output_data"])
    return program_config
