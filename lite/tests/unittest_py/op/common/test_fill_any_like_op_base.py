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
                min_value=1, max_value=10), min_size=1, max_size=4))
    value_data = draw(st.floats(min_value=-1, max_value=1))
    dtype = draw(st.sampled_from([-1, 2, 3, 5]))

    def generate_input_int32(*args, **kwargs):
        return np.random.random(in_shape).astype(np.int32)

    def generate_input_int64(*args, **kwargs):
        return np.random.random(in_shape).astype(np.int64)

    def generate_input_float32(*args, **kwargs):
        return np.random.random(in_shape).astype(np.float32)

    input_type = draw(
        st.sampled_from([
            generate_input_int32, generate_input_int64, generate_input_float32
        ]))

    fill_any_like_op = OpConfig(
        type="fill_any_like",
        inputs={"X": ["input_data"]},
        outputs={"Out": ["output_data"]},
        attrs={"value": value_data,
               "dtype": dtype})
    program_config = ProgramConfig(
        ops=[fill_any_like_op],
        weights={},
        inputs={"input_data": TensorConfig(data_gen=partial(input_type))},
        outputs=["output_data"])
    return program_config
