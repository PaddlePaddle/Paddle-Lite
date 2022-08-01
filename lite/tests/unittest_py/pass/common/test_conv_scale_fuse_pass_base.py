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
                min_value=1, max_value=64), min_size=4, max_size=4))
    paddings = draw(st.sampled_from([[1, 2], [4, 2]]))
    dilations = draw(st.sampled_from([[1, 1]]))
    groups = draw(st.sampled_from([1, 2, in_shape[1]]))
    padding_algorithm = draw(st.sampled_from(["VALID", "SAME"]))
    strides = draw(st.sampled_from([[1, 1], [2, 2]]))
    scale = draw(st.floats(min_value=0.5, max_value=5))
    scale_bias = draw(st.floats(min_value=0.0, max_value=1.0))

    assume(in_shape[1] == weight_shape[1] * groups)
    assume(weight_shape[0] % groups == 0)

    paddings_, dilations_ = UpdatePaddingAndDilation(
        in_shape, weight_shape, paddings, dilations, groups, padding_algorithm,
        strides)
    out_shape = [in_shape[0], weight_shape[0]]
    oh, ow = ConvOutputSize(in_shape, weight_shape, dilations_, paddings_,
                            strides)
    out_shape = out_shape + [oh, ow]

    assume(oh > 0 and ow > 0)

    conv_op = OpConfig(
        type="conv2d",
        inputs={
            "Input": ["input_data"],
            "Filter": ["weight_data"],
            "Bias": ["conv_bias"]
        },
        outputs={"Output": ["conv_output_data"]},
        attrs={
            "data_format": 'nchw',
            "dilations": dilations,
            "padding_algorithm": padding_algorithm,
            "groups": groups,
            "paddings": paddings,
            "strides": strides
        })

    scale_op = OpConfig(
        type="scale",
        inputs={"X": ["conv_output_data"]},
        outputs={"Out": ["output_data"]},
        attrs={"scale": scale,
               "bias": scale_bias,
               "bias_after_scale": True})

    ops = [conv_op, scale_op]
    program_config = ProgramConfig(
        ops=ops,
        weights={
            "conv_bias": TensorConfig(shape=[weight_shape[0]]),
            "weight_data": TensorConfig(shape=weight_shape)
        },
        inputs={"input_data": TensorConfig(shape=in_shape)},
        outputs=["output_data"])
    return program_config
