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
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=8), min_size=3, max_size=4))
    target1 = []
    target2 = []
    target3 = []
    for i in range(len(in_shape)):
        target1.append(in_shape[i] * (i + 1))
        target2.append(in_shape[i] * (i + 1) * 2)
        target3.append(in_shape[i] * (i + 1) * 3)
    target_shape = draw(st.sampled_from([target1, target2, target3]))

    def generate_input_int64(*args, **kwargs):
        return np.random.random(in_shape).astype(np.int64)

    def generate_input_float32(*args, **kwargs):
        return np.random.random(in_shape).astype(np.float32)

    input_type = draw(
        st.sampled_from([generate_input_int64, generate_input_float32]))

    def generate_target(*args, **kwargs):
        if input_type == generate_input_int64:
            return np.random.random(target_shape).astype(np.int64)
        else:
            return np.random.random(target_shape).astype(np.float32)

    expand_as_op = OpConfig(
        type="expand_as",
        inputs={"X": ["input_data"],
                "target_tensor": ["target_data"]},
        outputs={"Out": ["output_data"]},
        attrs={})

    program_config = ProgramConfig(
        ops=[expand_as_op],
        weights={},
        inputs={
            "input_data": TensorConfig(data_gen=partial(input_type)),
            "target_data": TensorConfig(data_gen=partial(generate_target))
        },
        outputs=["output_data"])
    return program_config
