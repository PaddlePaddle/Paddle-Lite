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
    bool_orimode = draw(st.sampled_from([True, False]))
    in_shape = draw(st.sampled_from([[5, 15], [10, 30], [25, 75], [40, 120]]))
    batch = draw(st.integers(min_value=1, max_value=10))

    def generate_input(*args, **kwargs):
        return np.random.random([batch, in_shape[1]]).astype(np.float32)

    def generate_weight(*args, **kwargs):
        return np.random.random(in_shape).astype(np.float32)

    def generate_hid(*args, **kwargs):
        return np.random.random([batch, in_shape[0]]).astype(np.float32)

    build_ops = OpConfig(
        type="gru_unit",
        inputs={
            "Input": ["input_data"],
            "HiddenPrev": ["hid_data"],
            "Weight": ["weight_data"]
        },
        outputs={
            "Gate": ["gate"],
            "ResetHiddenPrev": ["reset_hidden"],
            "Hidden": ["hidden"],
        },
        attrs={
            "activation": 2,  #tanh
            "gate_activation": 1,  #sigmoid
            "origin_mode": bool_orimode,
        })
    program_config = ProgramConfig(
        ops=[build_ops],
        weights={
            "weight_data": TensorConfig(data_gen=partial(generate_weight)),
        },
        inputs={
            "input_data": TensorConfig(
                data_gen=partial(generate_input), lod=[[0, 2, 6, 9]]),
            "hid_data": TensorConfig(data_gen=partial(generate_hid)),
        },
        outputs=["gate", "reset_hidden", "hidden"])
    return program_config
