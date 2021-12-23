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
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=8), min_size=4, max_size=4))

    update_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=8), min_size=4, max_size=4))
    assume(
        len(update_shape) == len(in_shape) and
        update_shape[1:] == in_shape[1:])

    index_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=8),
            min_size=update_shape[-1],
            max_size=update_shape[-1]))
    index_type = draw(st.sampled_from(["int32", "int64"]))
    overwrite = draw(st.booleans())

    def generate_update(*args, **kwargs):
        return np.random.randint(-10, 10, update_shape).astype(np.float32)

    def generate_index_int32(*args, **kwargs):
        return np.random.randint(-10, 10, index_shape).astype(np.int32)

    def generate_index_int64(*args, **kwargs):
        return np.random.randint(-10, 10, index_shape).astype(np.int64)

    def generate_input_float32(*args, **kwargs):
        return np.random.random(-1.0, 1.0, in_shape).astype(np.float32)

    scatter_op = OpConfig(
        type="scatter",
        inputs={
            "X": ["input_data"],
            "Ids": ["index"],
            "Updates": ["updates"]
        },
        outputs={"Out": ["output_data"]},
        attrs={"overwrite": overwrite})

    if index_type == "int32":
        program_config = ProgramConfig(
            ops=[scatter_op],
            weights={},
            inputs={
                "input_data":
                TensorConfig(data_gen=partial(generate_input_float32)),
                "index": TensorConfig(data_gen=partial(generate_index_int32)),
                "updates": TensorConfig(data_gen=partial(generate_update))
            },
            outputs=["output_data"])
    elif index_type == "int64":
        program_config = ProgramConfig(
            ops=[scatter_op],
            weights={},
            inputs={
                "input_data":
                TensorConfig(data_gen=partial(generate_input_float32)),
                "index": TensorConfig(data_gen=partial(generate_index_int64)),
                "updates": TensorConfig(data_gen=partial(generate_update))
            },
            outputs=["output_data"])

    return program_config
