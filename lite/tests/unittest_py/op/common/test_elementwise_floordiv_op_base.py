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
    input_data_x_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=8), min_size=1, max_size=8))
    axis = draw(
        st.integers(
            min_value=-1, max_value=(len(input_data_x_shape) - 1)))
    input_data_y_shape = input_data_x_shape[axis:]

    def gen_input_data_x():
        return np.random.randint(
            1, 3, size=(input_data_x_shape)).astype(np.int64)

    def gen_input_data_y():
        return np.random.randint(
            1, 3, size=(input_data_y_shape)).astype(np.int64)

    elementwise_floordiv_op = OpConfig(
        type="elementwise_floordiv",
        inputs={"X": ["input_data_x"],
                "Y": ["input_data_y"]},
        outputs={"Out": ["output_data"]},
        attrs={"axis": axis})
    program_config = ProgramConfig(
        ops=[elementwise_floordiv_op],
        weights={},
        inputs={
            "input_data_x": TensorConfig(data_gen=gen_input_data_x),
            "input_data_y": TensorConfig(data_gen=gen_input_data_y)
        },
        outputs=["output_data"])
    return program_config
