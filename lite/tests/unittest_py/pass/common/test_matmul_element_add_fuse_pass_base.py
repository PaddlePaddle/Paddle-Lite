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
    matmul_x_shape = draw(
        st.lists(
            st.integers(
                min_value=5, max_value=10), min_size=3, max_size=4))

    transpose_Y_data = draw(st.sampled_from([False, True]))
    matmul_y_shape = []
    matmul_y1 = draw(st.integers(min_value=5, max_value=10))
    if transpose_Y_data == False:
        matmul_y_shape = [matmul_x_shape[-1], matmul_y1]
    else:
        matmul_y_shape = [matmul_y1, matmul_x_shape[-1]]
    add_x_data_shape = [int(1), matmul_y1]

    matmul_op = OpConfig(
        type="matmul",
        inputs={"X": ["x_data"],
                "Y": ["y_data"]},
        outputs={"Out": ["matmul_output_data"]},
        attrs={
            "transpose_X": False,
            "transpose_Y": transpose_Y_data,
            "alpha": 1.0,
            "fused_reshape_X": [],
            "fused_reshape_Y": [],
            "fused_transpose_X": [],
            "fused_transpose_Y": []
        })

    elementwise_add_op = OpConfig(
        type="elementwise_add",
        inputs={"X": ["matmul_output_data"],
                "Y": ["add_x_data"]},
        outputs={"Out": ["output_data"]},
        attrs={"axis": -1})

    ops = [matmul_op, elementwise_add_op]

    weights_ = {
        "add_x_data": TensorConfig(shape=add_x_data_shape),
        "y_data": TensorConfig(shape=matmul_y_shape)
    }
    inputs_ = {"x_data": TensorConfig(shape=matmul_x_shape)}

    program_config = ProgramConfig(
        ops=ops, weights=weights_, inputs=inputs_, outputs=["output_data"])
    return program_config
