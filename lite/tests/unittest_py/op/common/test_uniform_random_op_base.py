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
    def generate_ShapeTensor():
        return np.random.randint(1, 5, size=[4]).astype(np.int64)

    shape_data = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=5), min_size=4, max_size=4))
    min_data = draw(st.floats(min_value=-1, max_value=-1))
    max_data = draw(st.floats(min_value=1, max_value=1))
    seed_data = draw(st.integers(min_value=0, max_value=0))
    dtype_data = draw(st.integers(min_value=5, max_value=5))  # out is float

    uniform_random_op = OpConfig(
        type="uniform_random",
        inputs={
            "ShapeTensor": ["ShapeTensor_data"],
            "ShapeTensorList": ["ShapeTensorList_data"]
        },
        outputs={"Out": ["output_data"]},
        attrs={
            "shape": shape_data,
            "min": min_data,
            "max": max_data,
            "seed": seed_data,
            "dtype": dtype_data,
            # lite does not use these 3 attr
            # so I default them
            "diag_num": 0,
            "diag_step": 0,
            "diag_val": 1.0,
        })
    program_config = ProgramConfig(
        ops=[uniform_random_op],
        weights={},
        inputs={
            "ShapeTensor_data":
            TensorConfig(data_gen=partial(generate_ShapeTensor)),
            "ShapeTensorList_data":
            TensorConfig(data_gen=partial(generate_ShapeTensor))
        },
        outputs=["output_data"])
    return program_config
