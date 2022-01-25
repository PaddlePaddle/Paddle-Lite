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
import copy


def sample_program_configs(draw):
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=10), min_size=1, max_size=2))

    def generate_input(*args, **kwargs):
        last_dim = np.random.randint(low=1, high=3, size=[1]).astype(np.int32)
        input_dim = copy.deepcopy(in_shape)
        input_dim.append(last_dim[0])  #last 2 dim must be equal
        input_dim.append(last_dim[0])
        return np.random.random(input_dim).astype(np.float32)

    build_ops = OpConfig(
        type="inverse",
        inputs={"Input": ["input_data"], },
        outputs={"Output": ["output_data"], },
        attrs={})

    program_config = ProgramConfig(
        ops=[build_ops],
        weights={},
        inputs={
            "input_data": TensorConfig(data_gen=partial(generate_input)),
        },
        outputs=["output_data"])
    return program_config
