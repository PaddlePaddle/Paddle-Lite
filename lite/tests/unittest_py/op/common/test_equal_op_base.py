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
    in_shape = draw(st.lists(st.integers(min_value=1, max_value=8), min_size = 1, max_size=6))
    equal_op = OpConfig(
        type = "equal",
        inputs = {"X" : ["input_data_x"],
                  "Y" : ["input_data_y"]},
        outputs = {"Out": ["output_data"]},
        attrs = {})
    cast_op = OpConfig(
        type = "cast",
        inputs = {"X" : ["output_data"]},
        outputs = {"Out": ["cast_output_data"]},
        attrs = {'in_dtype': 0,
                 'out_dtype': 5}) # 0: bool , 5: float
    program_config = ProgramConfig(
        ops=[equal_op, cast_op],
        weights={},
        inputs={ "input_data_x": TensorConfig(shape=in_shape),
                 "input_data_y": TensorConfig(shape=in_shape)
        },
        outputs=["cast_output_data"])
    return program_config
