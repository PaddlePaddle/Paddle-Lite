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
    in_shape = draw(st.lists(
        st.integers(
            min_value=1, max_value=64), min_size=1, max_size=4))
    input_type = draw(st.sampled_from(["type_float", "type_int", "type_int64"]))
    input_axis = draw(st.sampled_from([-2, -1, 0, 1, 2, 3]))
    axis = input_axis if input_axis > 0 else input_axis + 1
    assume(abs(axis) < len(in_shape))
    if(input_type != "type_float"):
        assume(abs(axis) <= 1)
    
    def generate_input1(*args, **kwargs):
        if input_type == "type_float":
            return np.random.normal(0.0, 1.0, in_shape).astype(np.float32)
        elif input_type == "type_int":
            return np.random.randint(in_shape).astype(np.int32)
        elif input_type == "type_int64":
            return np.random.randint(in_shape).astype(np.int64)
    
    def generate_input2(*args, **kwargs):
        if input_type == "type_float":
            return np.random.normal(0.0, 1.0, in_shape).astype(np.float32)
        elif input_type == "type_int":
            return np.random.randint(in_shape).astype(np.int32)
        elif input_type == "type_int64":
            return np.random.randint(in_shape).astype(np.int64)

    def generate_input3(*args, **kwargs):
        if input_type == "type_float":
            return np.random.normal(0.0, 1.0, in_shape).astype(np.float32)
        elif input_type == "type_int":
            return np.random.randint(in_shape).astype(np.int32)
        elif input_type == "type_int64":
            return np.random.randint(in_shape).astype(np.int64)

    ops_config = OpConfig(
        type = "stack",
        inputs = {
            "X":["stack_input1", "stack_input2", "stack_input3"]
        },
        outputs = {
            "Y": ["output_data"]
        },
        attrs = {
            "axis": input_axis
        }
        )

    program_config = ProgramConfig(
        ops=[ops_config],
        weights={},
        inputs={
            "stack_input1": 
            TensorConfig(data_gen=partial(generate_input1)),
            "stack_input2": 
            TensorConfig(data_gen=partial(generate_input2)),
            "stack_input3": 
            TensorConfig(data_gen=partial(generate_input3))
        },
        outputs=["output_data"])

    return program_config
