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
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


def sample_program_configs(draw):
    alpha = draw(st.floats(min_value=1, max_value=1))  #required in pass
    x_num_col_dims = draw(st.floats(min_value=0, max_value=1))
    y_num_col_dims = draw(st.floats(min_value=0, max_value=1))
    int32_values_1 = draw(st.integers(min_value=1, max_value=40))
    int32_values_2 = draw(st.integers(min_value=1, max_value=40))
    int32_values_3 = draw(st.integers(min_value=1, max_value=40))

    squeeze2_input_shape = [int32_values_1, int32_values_2, 1, 1]
    matmul_input_shape = [squeeze2_input_shape[1], int32_values_3]

    squeeze2_op = OpConfig(
        type="squeeze2",
        inputs={"X": ["squeeze2_input_x"]},
        outputs={
            "Out": ["squeeze2_output"],
            "XShape": ["squeeze2_output_XShape"]
        },
        attrs={
            "data_format": 'nchw',
            "axes": [2, 3]  #required in pass
        })

    matmul_op = OpConfig(
        type="matmul",
        inputs={"X": ["squeeze2_output"],
                "Y": ["matmul_input"]},
        outputs={"Out": ["output_data"]},
        attrs={
            "transpose_X": False,
            "transpose_Y": False,
            "x_num_col_dims": x_num_col_dims,
            "y_num_col_dims": y_num_col_dims,
            "alpha": alpha,
            "fused_reshape_X": [],
            "fused_transpose_X": [],
            "fused_reshape_Y": [],
            "fused_transpose_Y": []
        })

    ops = [squeeze2_op, matmul_op]
    program_config = ProgramConfig(
        ops=ops,
        weights={},
        inputs={
            "squeeze2_input_x": TensorConfig(shape=squeeze2_input_shape),
            "matmul_input": TensorConfig(shape=matmul_input_shape)
        },
        outputs=["output_data"])
    return program_config
