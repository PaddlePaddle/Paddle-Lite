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
    shape0 = draw(st.integers(min_value=1, max_value=64))
    shape1 = draw(st.integers(min_value=1, max_value=64))
    shape2 = draw(st.integers(min_value=1, max_value=64))
    batch0 = draw(st.integers(min_value=1, max_value=64))
    batch1 = draw(st.integers(min_value=1, max_value=64))
    transpose_X = draw(st.booleans())
    transpose_Y = draw(st.booleans())
    if ((not transpose_X) and (not transpose_Y)):
        X_shape = [batch0, 1, shape0, shape1]
        Y_shape = [batch0, shape1, shape2]
    if ((transpose_X) and (not transpose_Y)):
        X_shape = [batch1, 1, shape1, shape0]
        Y_shape = [batch1, 1, shape1, shape2]
    if ((not transpose_X) and (transpose_Y)):
        X_shape = [batch0, shape0, shape1]
        Y_shape = [batch0, shape2, shape1]
    if ((transpose_X) and (transpose_Y)):
        X_shape = [batch0, shape1, shape0]
        Y_shape = [batch0, shape2, shape1]

    matmul_v2_op = OpConfig(
        type="matmul_v2",
        inputs={"X": ["input_data_x"],
                "Y": ["input_data_y"]},
        outputs={"Out": ["output_data"]},
        attrs={"trans_x": transpose_X,
               "trans_y": transpose_Y})
    program_config = ProgramConfig(
        ops=[matmul_v2_op],
        weights={},
        inputs={
            "input_data_x": TensorConfig(shape=X_shape),
            "input_data_y": TensorConfig(shape=Y_shape)
        },
        outputs={"output_data"})
    return program_config
