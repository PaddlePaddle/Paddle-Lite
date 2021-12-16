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
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


def sample_program_configs(draw, elementwise_type):
    in_shape_x = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=20), min_size=4, max_size=4))
    in_shape_y = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=20), min_size=4, max_size=4))
    assume((in_shape_x[0] == in_shape_y[0] or in_shape_x[0] == 1 or
            in_shape_y[0] == 1) and (in_shape_x[0] >= in_shape_y[0]))
    assume((in_shape_x[1] == in_shape_y[1] or in_shape_x[1] == 1 or
            in_shape_y[1] == 1) and (in_shape_x[1] >= in_shape_y[1]))
    assume((in_shape_x[2] == in_shape_y[2] or in_shape_x[2] == 1 or
            in_shape_y[2] == 1) and (in_shape_x[2] >= in_shape_y[2]))
    assume((in_shape_x[3] == in_shape_y[3] or in_shape_x[3] == 1 or
            in_shape_y[3] == 1) and (in_shape_x[3] >= in_shape_y[3]))

    axis = -1
    elementwise_op = OpConfig(
        type=elementwise_type,
        inputs={"X": ["input_data_x"],
                "Y": ["input_data_y"]},
        outputs={"Out": ["elementwise_output_data"]},
        attrs={"data_format": 'nchw',
               "axis": axis})

    act_type = draw(st.sampled_from(['relu']))

    def generate_act_attrs(act_type_str):
        attrs = {}
        if act_type_str == 'relu':
            attrs = {}
        return attrs

    active_op = OpConfig(
        type=act_type,
        inputs={"X": ["elementwise_output_data"]},
        outputs={"Out": ["output_data"]},
        attrs=generate_act_attrs(act_type))

    ops = [elementwise_op, active_op]
    program_config = ProgramConfig(
        ops=ops,
        weights={},
        inputs={
            "input_data_x": TensorConfig(shape=in_shape_x),
            "input_data_y": TensorConfig(shape=in_shape_y)
        },
        outputs=["output_data"])
    return program_config
