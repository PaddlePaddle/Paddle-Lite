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
                min_value=4, max_value=5), min_size=4, max_size=4))
    repeat_times_data = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=5), min_size=4, max_size=4))

    choose_repeat = draw(
        st.sampled_from(
            ["RepeatTimes", "repeat_times_tensor", "repeat_times"]))

    def generate_RepeatTimes_data():
        if (choose_repeat == "RepeatTimes"):
            return np.array(repeat_times_data).astype(np.int32)
        else:
            return np.random.randint(1, 5, []).astype(np.int32)

    def generate_repeat_times_tensor_data():
        if (choose_repeat == "repeat_times_tensor"):
            return np.array(repeat_times_data).astype(np.int32)
        else:
            return np.random.randint(1, 5, []).astype(np.int32)

    tile_op = OpConfig(
        type="tile",
        inputs={
            "X": ["X_data"],
            "RepeatTimes": ["RepeatTimes_data"],
            "repeat_times_tensor": ["repeat_times_tensor_data"]
        },
        outputs={"Out": ["Out_data"]},
        attrs={"repeat_times": repeat_times_data})
    program_config = ProgramConfig(
        ops=[tile_op],
        weights={},
        inputs={
            "X_data": TensorConfig(shape=in_shape),
            "RepeatTimes_data":
            TensorConfig(data_gen=partial(generate_RepeatTimes_data)),
            "repeat_times_tensor_data":
            TensorConfig(data_gen=partial(generate_repeat_times_tensor_data)),
        },
        outputs=["Out_data"])

    return program_config
