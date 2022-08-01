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
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st


def sample_program_configs(draw):
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=64), min_size=4, max_size=4))
    group = draw(st.integers(min_value=1, max_value=10))

    assume(in_shape[1] % group == 0)

    def generate_input(*args, **kwargs):
        return np.random.normal(0.0, 1.0, in_shape).astype(np.float32)

    ops_config = OpConfig(
        type="shuffle_channel",
        inputs={"X": ["input_data"]},
        outputs={"Out": ["output_data"]},
        attrs={"group": group})

    program_config = ProgramConfig(
        ops=[ops_config],
        weights={},
        inputs={"input_data": TensorConfig(data_gen=partial(generate_input))},
        outputs=["output_data"])

    return program_config
