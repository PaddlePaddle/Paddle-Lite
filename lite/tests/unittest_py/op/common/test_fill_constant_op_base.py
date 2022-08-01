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
import hypothesis.strategies as st
import random


def sample_program_configs(draw):
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=10), min_size=1, max_size=4))
    tensor_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=10), min_size=1, max_size=4))
    dtype = draw(st.sampled_from([2, 3, 5]))

    with_value_tensor = draw(st.sampled_from([True, False]))
    with_shape_tensor = draw(st.sampled_from([True, False]))

    def generate_shape_tensor(*args, **kwargs):
        return np.array(tensor_shape).astype(np.int32)

    def generate_input_bool(*args, **kwargs):
        return np.random.random([1]).astype(np.bool)

    def generate_input_int8(*args, **kwargs):
        return np.random.random([1]).astype(np.int8)

    def generate_input_int32(*args, **kwargs):
        return np.random.random([1]).astype(np.int32)

    def generate_input_int64(*args, **kwargs):
        return np.random.random([1]).astype(np.int64)

    def generate_input_float32(*args, **kwargs):
        return np.random.random([1]).astype(np.float32)

    if dtype == 0:
        value_data = generate_input_bool
    elif dtype == 2:
        value_data = generate_input_int32
    elif dtype == 3:
        value_data = generate_input_int64
    elif dtype == 5:
        value_data = generate_input_float32
    else:
        value_data = generate_input_int8

    value = draw(st.floats(min_value=-10, max_value=10))
    op_inputs = {}
    program_inputs = {}

    #ShapeTensorList not support now 
    if (with_value_tensor and with_shape_tensor):
        op_inputs = {
            "ValueTensor": ["value_data"],
            "ShapeTensor": ["shape_data"]
        }
        program_inputs = {
            "value_data": TensorConfig(data_gen=partial(value_data)),
            "shape_data": TensorConfig(data_gen=partial(generate_shape_tensor))
        }
    elif ((not with_value_tensor) and with_shape_tensor):
        op_inputs = {"ShapeTensor": ["shape_data"]}
        program_inputs = {
            "shape_data": TensorConfig(data_gen=partial(generate_shape_tensor))
        }
    elif (with_value_tensor and (not with_shape_tensor)):
        op_inputs = {"ValueTensor": ["value_data"]}
        program_inputs = {
            "value_data": TensorConfig(data_gen=partial(value_data))
        }

    fill_constant_op = OpConfig(
        type="fill_constant",
        inputs=op_inputs,
        outputs={"Out": ["output_data"]},
        attrs={
            "dtype": dtype,
            "shape": in_shape,
            "value": value,
            "force_cpu": False
            #"place_type" : -1
        })
    program_config = ProgramConfig(
        ops=[fill_constant_op],
        weights={},
        inputs=program_inputs,
        outputs=["output_data"])
    return program_config
