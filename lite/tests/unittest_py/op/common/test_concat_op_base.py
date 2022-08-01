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
    in_shape1 = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=10), min_size=1, max_size=4))
    in_shape2 = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=10), min_size=1, max_size=4))
    axis = draw(st.sampled_from([-1, 0, 1, 2, 3]))

    def generate_input1(*args, **kwargs):
        return np.random.random(in_shape1).astype(np.float32)

    def generate_input2(*args, **kwargs):
        return np.random.random(in_shape2).astype(np.float32)

    def generate_axis(*args, **kwargs):
        return np.array([axis]).astype("int32")

    concat_op = OpConfig(
        type="concat",
        inputs={
            "X": ["input_data1", "input_data2"],
            "AxisTensor": ["axis_tensor_data"]
        },
        outputs={"Out": ["output_data"]},
        attrs={"axis": axis})
    program_config = ProgramConfig(
        ops=[concat_op],
        weights={},
        inputs={
            "input_data1": TensorConfig(data_gen=partial(generate_input1)),
            "input_data2": TensorConfig(data_gen=partial(generate_input2)),
            "axis_tensor_data": TensorConfig(data_gen=partial(generate_axis)),
        },
        outputs=["output_data"])
    return program_config
