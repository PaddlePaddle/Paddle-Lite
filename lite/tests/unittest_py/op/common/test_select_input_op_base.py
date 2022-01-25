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
    def generate_input(*args, **kwargs):
        if kwargs["type"] == "int32":
            return np.random.randint(kwargs["low"], kwargs["high"],
                                     kwargs["shape"]).astype(np.int32)
        elif kwargs["type"] == "int64":
            return np.random.randint(kwargs["low"], kwargs["high"],
                                     kwargs["shape"]).astype(np.int64)
        elif kwargs["type"] == "float32":
            return np.random.random(kwargs["shape"]).astype(np.float32)

    input_type = draw(st.sampled_from(["int32", "int64", "float32"]))

    max_dim_size = 4
    x1_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=8),
            min_size=4,
            max_size=max_dim_size))
    x2_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=8),
            min_size=4,
            max_size=max_dim_size))
    x3_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=8),
            min_size=4,
            max_size=max_dim_size))
    mask_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=1), min_size=1, max_size=3))

    select_input_op = OpConfig(
        type="select_input",
        inputs={"X": ["x1_data", "x2_data", "x3_data"],
                "Mask": ["mask"]},
        outputs={"Out": ["output_data"]},
        attrs={})

    program_config = ProgramConfig(
        ops=[select_input_op],
        weights={},
        inputs={
            "x1_data": TensorConfig(data_gen=partial(
                generate_input,
                type=input_type,
                low=-10,
                high=10,
                shape=x1_shape)),
            "x2_data": TensorConfig(data_gen=partial(
                generate_input,
                type=input_type,
                low=-10,
                high=10,
                shape=x2_shape)),
            "x3_data": TensorConfig(data_gen=partial(
                generate_input,
                type=input_type,
                low=-10,
                high=10,
                shape=x3_shape)),
            "mask": TensorConfig(data_gen=partial(
                generate_input, type="int32", low=1, high=2, shape=mask_shape))
        },
        outputs=["output_data"])

    return program_config
