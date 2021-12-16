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
from functools import partial, reduce
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


def mul(x, y):
    return x * y


def sample_program_configs(draw):
    mul_x_in_shape = draw(
        st.lists(
            st.integers(
                min_value=5, max_value=10), min_size=3, max_size=5))
    x_num_col_dims_data = len(mul_x_in_shape)
    last = draw(st.integers(min_value=5, max_value=10))
    mul_x_in_shape = mul_x_in_shape + [last]
    #lite not check fuse condition : bias[0]=1 bias[1]=weight[1] 
    add_x_data_shape = [1, draw(st.integers(min_value=5, max_value=10))]

    mul_op = OpConfig(
        type="mul",
        inputs={"X": ["mul_x_data"],
                "Y": ["mul_y_data"]},
        outputs={"Out": ["mul_output_data"]},
        attrs={"x_num_col_dims": x_num_col_dims_data,
               "y_num_col_dims": 1})

    elementwise_add_op = OpConfig(
        type="elementwise_add",
        inputs={"X": ["mul_output_data"],
                "Y": ["add_x_data"]},
        outputs={"Out": ["output_data"]},
        attrs={"axis": -1})

    ops = [mul_op, elementwise_add_op]
    program_config = ProgramConfig(
        ops=ops,
        weights={"add_x_data": TensorConfig(shape=add_x_data_shape)},
        inputs={
            "mul_x_data": TensorConfig(shape=mul_x_in_shape),
            "mul_y_data": TensorConfig(shape=[last, add_x_data_shape[1]])
        },
        outputs=["output_data"])
    return program_config
