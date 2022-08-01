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
from hypothesis import assume


def sample_program_configs(draw):
    #The number of elements in Input(X) should be 1
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=1), min_size=1, max_size=1))
    step_data = draw(st.floats(min_value=0.1, max_value=0.5))
    input_type = draw(
        st.sampled_from(["type_int", "type_int64", "type_float"]))

    def generate_input1(*args, **kwargs):
        return np.random.random(in_shape).astype(np.float32)

    def generate_input2(*args, **kwargs):
        return np.random.randint(in_shape).astype(np.int32)

    def generate_input3(*args, **kwargs):
        return np.random.randint(in_shape).astype(np.int64)

    build_ops = OpConfig(
        type="increment",
        inputs={"X": ["input_data"], },
        outputs={"Out": ["output_data"], },
        attrs={"step": step_data, })
    if input_type == "type_int":
        program_config = ProgramConfig(
            ops=[build_ops],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input2)),
            },
            outputs=["output_data"])
    elif input_type == "type_int64":
        program_config = ProgramConfig(
            ops=[build_ops],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input3)),
            },
            outputs=["output_data"])
    elif input_type == "type_float":
        program_config = ProgramConfig(
            ops=[build_ops],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input1)),
            },
            outputs=["output_data"])

    return program_config
