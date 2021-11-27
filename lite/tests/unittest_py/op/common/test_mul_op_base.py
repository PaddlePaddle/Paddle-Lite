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
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st

def sample_program_configs(draw):
    in_shape=draw(st.lists(
        st.integers(
            min_value=2, max_value=2), min_size=2, max_size=2))

    mul_op = OpConfig(
        type = "mul",
        inputs = {"X": ["input_data_x"],
                    "Y": ["input_data_y"]},
        outputs = {"Out": ["output_data"]},
        attrs = {"x_num_col_dims": 1,
                    "y_num_col_dims": 1})

    program_config = ProgramConfig(
        ops=[mul_op],
        weights={
            "input_data_y":
             TensorConfig(shape=in_shape)
        },
        inputs={
            "input_data_x":
            TensorConfig(shape=in_shape)
        },
        outputs=["output_data"])

    return program_config
