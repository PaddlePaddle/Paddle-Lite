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
    in_shape0 = draw(
        st.lists(
            st.integers(
                min_value=3, max_value=64), min_size=4, max_size=4))
    weight_shape0 = [
        draw(st.integers(
            min_value=3, max_value=64)), in_shape0[1], 1, 1
    ]
    weight_shape1 = [
        draw(st.integers(
            min_value=3, max_value=64)), weight_shape0[0], 1, 1
    ]

    assume(in_shape0[1] == weight_shape0[1])
    mula = in_shape0[1] * (weight_shape1[0] - weight_shape0[0])
    mulb = weight_shape0[0] * weight_shape1[0]
    assume(not (mula <= 0 or mula > mulb))

    paddings_, dilations_ = UpdatePaddingAndDilation(
        in_shape=in_shape0,
        weight_shape=weight_shape0,
        paddings=[0, 0],
        dilations=[1, 1],
        groups=1,
        padding_algorithm="VALID",
        strides=[1, 1])
    out_shape = [in_shape0[0], weight_shape0[0]]
    oh, ow = ConvOutputSize(
        in_shape=in_shape0,
        weight_shape=weight_shape0,
        dilations=dilations_,
        paddings=paddings_,
        strides=[1, 1])
    out_shape0 = out_shape + [oh, ow]
    assume(oh > 0 and ow > 0)

    in_shape1 = out_shape0
    paddings_, dilations_ = UpdatePaddingAndDilation(
        in_shape=in_shape1,
        weight_shape=weight_shape1,
        paddings=[0, 0],
        dilations=[1, 1],
        groups=1,
        padding_algorithm="VALID",
        strides=[1, 1])
    out_shape = [in_shape1[0], weight_shape1[0]]
    oh, ow = ConvOutputSize(
        in_shape=in_shape1,
        weight_shape=weight_shape1,
        dilations=dilations_,
        paddings=paddings_,
        strides=[1, 1])
    out_shape1 = out_shape + [oh, ow]
    assume(oh > 0 and ow > 0)

    conv0_op = OpConfig(
        type="conv2d",
        inputs={"Input": ["input_data"],
                "Filter": ["weight_data0"]},
        outputs={"Output": ["conv_output_data"]},
        attrs={
            "data_format": 'nchw',
            "dilations": [1, 1],
            "padding_algorithm": "VALID",
            "groups": 1,
            "paddings": [0, 0],
            "strides": [1, 1]
        })

    conv1_op = OpConfig(
        type="conv2d",
        inputs={"Input": ["conv_output_data"],
                "Filter": ["weight_data1"]},
        outputs={"Output": ["output_data"]},
        attrs={
            "data_format": 'nchw',
            "dilations": [1, 1],
            "padding_algorithm": "VALID",
            "groups": 1,
            "paddings": [0, 0],
            "strides": [1, 1]
        })

    ops = [conv0_op, conv1_op]
    program_config = ProgramConfig(
        ops=ops,
        weights={
            "weight_data0": TensorConfig(shape=weight_shape0),
            "weight_data1": TensorConfig(shape=weight_shape1)
        },
        inputs={"input_data": TensorConfig(shape=in_shape0), },
        outputs=["output_data"])
    return program_config
