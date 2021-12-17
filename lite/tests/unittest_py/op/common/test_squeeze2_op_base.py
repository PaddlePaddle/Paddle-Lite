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
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=10), min_size=0, max_size=3))
    in_shape.insert(0, 1)
    input_type = draw(st.sampled_from(["type_float", "type_int"]))
    #input_type = draw(st.sampled_from(["type_float", "type_int", "type_int64"]))
    input_axis = draw(
        st.sampled_from([[], [-1], [0], [1], [2], [3], [-1, 0], [0, 1],
                         [2, 3], [0, 1, 2, 3]]))
    assume(len(input_axis) <= len(in_shape))
    if len(input_axis) > 0:
        for num in input_axis:
            assume(num < len(in_shape))
            assume(in_shape[num] == 1)
            if (input_type != "type_float"):
                assume(num <= 0)

    def generate_input(*args, **kwargs):
        if input_type == "type_float":
            return np.random.normal(0.0, 1.0, in_shape).astype(np.float32)
        elif input_type == "type_int":
            return np.random.randint(in_shape).astype(np.int32)
        elif input_type == "type_int64":
            return np.random.randint(in_shape).astype(np.int64)

    def generate_xshape(*args, **kwargs):
        if input_type == "type_float":
            return np.random.normal(0.0, 1.0, in_shape).astype(np.float32)
        elif input_type == "type_int":
            return np.random.randint(in_shape).astype(np.int32)
        elif input_type == "type_int64":
            return np.random.randint(in_shape).astype(np.int64)

    ops_config = OpConfig(
        type="squeeze2",
        inputs={"X": ["input_data"]},
        outputs={"Out": ["output_data"],
                 "XShape": ["squeeze2_xshape"]},
        attrs={"axes": input_axis})

    program_config = ProgramConfig(
        ops=[ops_config],
        weights={
            "squeeze2_xshape": TensorConfig(data_gen=partial(generate_xshape))
        },
        inputs={"input_data": TensorConfig(data_gen=partial(generate_input))},
        outputs=["output_data"])

    return program_config
