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


# having diff !
def sample_program_configs(draw):
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=5), min_size=4, max_size=4))

    def generate_K_data():
        return np.random.randint(1, 3, size=[1]).astype(np.int32)

    k_data = draw(st.integers(min_value=1, max_value=2))
    axis_data = draw(st.integers(min_value=0, max_value=3))
    # Lite does not have these two attributes
    largest_data = draw(st.booleans())
    sorted_data = draw(st.booleans())

    assume(k_data <= in_shape[-1])

    top_k_v2_op = OpConfig(
        type="top_k_v2",
        inputs={
            "X": ["X_data"],
            #"K": ["K_data"]
        },
        outputs={"Out": ["Out_data"],
                 "Indices": ["Indices_data"]},
        attrs={
            "k": k_data,
            "axis": axis_data,
            #"largest": largest_data,
            #"sorted": sorted_data,
        })
    program_config = ProgramConfig(
        ops=[top_k_v2_op],
        weights={},
        inputs={
            "X_data": TensorConfig(shape=in_shape),
            "K_data": TensorConfig(data_gen=partial(generate_K_data))
        },
        outputs=["Out_data", "Indices_data"])

    return program_config
