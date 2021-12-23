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
    x_dim0 = draw(st.integers(min_value=1, max_value=100))
    x_dim1 = draw(st.integers(min_value=1, max_value=100))
    y_dim1 = draw(st.integers(min_value=1, max_value=100))

    matmul_op = OpConfig(
        type="matmul",
        inputs={"X": ["x_data"],
                "Y": ["y_data"]},
        outputs={"Out": ["output_data"]},
        attrs={
            "transpose_X": False,
            "transpose_Y": False,
            "alpha": 1.0,
            "fused_reshape_X": [],
            "fused_reshape_Y": [],
            "fused_transpose_X": [],
            "fused_transpose_Y": []
        })

    ops = [matmul_op]
    program_config = ProgramConfig(
        ops=ops,
        weights={},
        inputs={
            "x_data": TensorConfig(shape=[x_dim0, x_dim1]),
            "y_data": TensorConfig(shape=[x_dim1, y_dim1])
        },
        outputs=["output_data"])
    return program_config
