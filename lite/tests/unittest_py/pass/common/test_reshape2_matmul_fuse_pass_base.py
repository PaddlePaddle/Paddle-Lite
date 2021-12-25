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
    alpha = draw(st.floats(min_value=1, max_value=1))
    x_num_col_dims = draw(st.integers(min_value=0, max_value=0))
    y_num_col_dims = draw(st.integers(min_value=0, max_value=0))
    int32_values = draw(st.integers(min_value=1, max_value=64))
    dim2_values = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=40), min_size=2, max_size=2))
    dim4_values = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=20), min_size=4, max_size=4))
    assume(
        (dim2_values[0] * dim2_values[1]) == (dim4_values[0] * dim4_values[1]))
    assume((dim4_values[2] == 1) and (dim4_values[3] == 1))

    matmul_y_shape = [dim2_values[1], int32_values]
    reshape2_op = OpConfig(
        type="reshape2",
        inputs={"X": ["input_data_x"]},
        outputs={
            "Out": ["reshape2_output"],
            "XShape": ["reshape2_output_XShape"]
        },
        attrs={
            "data_format": 'nchw',
            "shape": dim2_values  #compare input_data_x->shape
        })

    matmul_op = OpConfig(
        type="matmul",
        inputs={"X": ["reshape2_output"],
                "Y": ["input_data_y"]},
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

    ops = [reshape2_op, matmul_op]
    program_config = ProgramConfig(
        ops=ops,
        weights={},
        inputs={
            "input_data_x": TensorConfig(shape=dim4_values),
            "input_data_y": TensorConfig(shape=matmul_y_shape)
        },
        outputs=["output_data"])
    return program_config
