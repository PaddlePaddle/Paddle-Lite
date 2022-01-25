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
                min_value=1, max_value=10), min_size=4, max_size=4))

    def generate_Condition_data():
        return np.random.choice([True, False], in_shape, replace=True)

    where_index_op = OpConfig(
        type="where_index",
        inputs={"Condition": ["Condition_data"]},
        outputs={"Out": ["Out_data"]},
        attrs={})
    program_config = ProgramConfig(
        ops=[where_index_op],
        weights={},
        inputs={
            "Condition_data":
            TensorConfig(data_gen=partial(generate_Condition_data))
        },
        outputs=["Out_data"])

    return program_config
