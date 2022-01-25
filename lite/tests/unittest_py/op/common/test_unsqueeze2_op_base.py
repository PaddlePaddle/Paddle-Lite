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
                min_value=1, max_value=5), min_size=4, max_size=4))
    axes_data = draw(
        st.lists(
            st.integers(
                min_value=0, max_value=3), min_size=1, max_size=2))

    def generate_AxesTensor_data():
        return np.random.choice([0, 1, 2, 3], axes_data, replace=True)

    def generate_AxesTensorList_data():
        return np.random.choice([0, 1, 2, 3], [], replace=True)

    def generate_XShape_data():
        return np.random.random([6]).astype(np.float32)

    unsqueeze2_op = OpConfig(
        type="unsqueeze2",
        inputs={
            "X": ["X_data"],
            "AxesTensor": ["AxesTensor_data"],
            "AxesTensorList": ["AxesTensorList_data"]
        },
        outputs={"Out": ["Out_data"],
                 "XShape": ["XShape_data"]},
        attrs={"axes": axes_data})
    program_config = ProgramConfig(
        ops=[unsqueeze2_op],
        weights={
            "XShape_data": TensorConfig(data_gen=partial(generate_XShape_data))
        },
        inputs={
            "X_data": TensorConfig(shape=in_shape),
            "AxesTensor_data":
            TensorConfig(data_gen=partial(generate_AxesTensor_data)),
            "AxesTensorList_data":
            TensorConfig(data_gen=partial(generate_AxesTensorList_data)),
        },
        outputs=["Out_data", "XShape_data"])
    return program_config
