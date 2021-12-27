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
from hypothesis import assume
import hypothesis.strategies as st


def sample_program_configs(draw):
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=5), min_size=4, max_size=4))
    axis_data = draw(st.integers(min_value=0, max_value=3))
    num_data = draw(st.integers(min_value=1, max_value=2))

    # declare multiple outputs
    output_string = ["out"] * num_data
    for i in range(num_data):
        output_string[i] += str(i)

    assume(num_data == in_shape[axis_data])

    unstack_op = OpConfig(
        type="unstack",
        inputs={"X": ["input_data"]},
        outputs={"Y": output_string},
        attrs={"axis": axis_data,
               "num": num_data})
    program_config = ProgramConfig(
        ops=[unstack_op],
        weights={},
        inputs={"input_data": TensorConfig(shape=in_shape), },
        outputs=output_string)
    return program_config
