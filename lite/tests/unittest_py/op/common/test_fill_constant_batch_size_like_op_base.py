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
                min_value=1, max_value=10), min_size=2, max_size=4))
    shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=10), min_size=2, max_size=4))
    dtype = draw(st.sampled_from([2, 3, 5]))
    input_dim_idx = draw(
        st.integers(
            min_value=0, max_value=(len(in_shape) - 1)))
    out_dim_idx = draw(st.integers(min_value=0, max_value=(len(shape) - 1)))
    value = draw(st.floats(min_value=-10, max_value=10))

    fill_constant_batch_size_like_op = OpConfig(
        type="fill_constant_batch_size_like",
        inputs={"Input": ["input_data"]},
        outputs={"Out": ["output_data"]},
        attrs={
            "dtype": dtype,
            "shape": shape,
            "value": value,
            "force_cpu": False,
            "input_dim_idx": input_dim_idx,
            "output_dim_idx": out_dim_idx
        })
    program_config = ProgramConfig(
        ops=[fill_constant_batch_size_like_op],
        weights={},
        inputs={"input_data": TensorConfig(shape=in_shape)},
        outputs=["output_data"])
    return program_config
