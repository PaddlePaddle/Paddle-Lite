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


def sample_program_configs(draw):
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=8), max_size=1))
    sub_block = draw(st.integers(min_value=1, max_value=8))
    is_scalar_condition = draw(st.booleans())

    def generate_input(*args, **kwargs):
        return np.random.random(in_shape).astype(np.float32)

    def generate_cond(*args, **kwargs):
        return np.random.random(in_shape).astype(np.bool)

    conditional_block_op = OpConfig(
        type="conditional_block",
        inputs={
            "Input": ["input_data"],
            "Cond": ["cond_data"],
        },
        outputs={"Out": ["output_data"],
                 "Scope": ["scope_data"]},
        attrs={
            "is_scalar_condition": is_scalar_condition,
            "sub_block": sub_block
        })

    program_config = ProgramConfig(
        ops=[conditional_block_op],
        weights={},
        inputs={
            "input_data": TensorConfig(data_gen=partial(generate_input)),
            "cond_data": TensorConfig(data_gen=partial(generate_cond))
        },
        outputs=["output_data", "scope_data"])
    return program_config
