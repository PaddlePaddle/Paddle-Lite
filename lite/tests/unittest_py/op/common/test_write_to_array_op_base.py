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
                min_value=2, max_value=6), min_size=4, max_size=4))
    axis_data = draw(st.integers(min_value=0, max_value=3))
    use_stack_data = draw(st.booleans())

    def generate_input_I_data():
        return np.random.randint(0, 1, [1]).astype(np.int64)

    write_to_array_op = OpConfig(
        type="write_to_array",
        inputs={"X": ["input_data"],
                "I": ["I_data"]},
        outputs={"Out": ["middle_data"]},
        attrs={})
    tensor_array_to_tensor_op = OpConfig(
        type="tensor_array_to_tensor",
        inputs={"X": ["middle_data"]},
        outputs={"Out": ["output_data"],
                 "OutIndex": ["OutIndex_data"]},
        attrs={
            "axis": axis_data,
            "use_stack": use_stack_data,
        })

    program_config = ProgramConfig(
        ops=[write_to_array_op, tensor_array_to_tensor_op],
        weights={},
        inputs={
            "input_data": TensorConfig(shape=in_shape),
            "I_data": TensorConfig(data_gen=partial(generate_input_I_data))
        },
        outputs=["output_data", "OutIndex_data"])
    return program_config
