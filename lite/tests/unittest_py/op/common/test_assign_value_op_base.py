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
import numpy


def sample_program_configs(draw):
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=4), min_size=1, max_size=4))
    dtype = draw(st.sampled_from([2, 3, 5]))
    bool_values = draw(st.lists(st.booleans(), min_size=1, max_size=4))
    fp32_values = draw(
        st.lists(
            st.floats(
                min_value=1, max_value=4), min_size=1, max_size=4))
    int32_values = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=4), min_size=1, max_size=4))
    int64_values = np.random.random([1]).astype(np.int64).tolist()
    in_shape_num = 1
    for val in in_shape:
        in_shape_num *= val
    if dtype == 2:
        assume(in_shape_num == len(int32_values))
    if dtype == 3:
        assume(in_shape_num == len(int64_values))
    if dtype == 5:
        assume(in_shape_num == len(fp32_values))

    assign_value_op = OpConfig(
        type="assign_value",
        inputs={},
        outputs={"Out": ["output_data"]},
        attrs={
            "shape": in_shape,
            "dtype": dtype,
            "bool_values": bool_values,
            "fp32_values": fp32_values,
            "int32_values": int32_values,
            "int64_values": int64_values
        })

    program_config = ProgramConfig(
        ops=[assign_value_op], weights={}, inputs={}, outputs=["output_data"])
    return program_config
