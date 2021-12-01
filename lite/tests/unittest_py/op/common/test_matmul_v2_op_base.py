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
    X_shape = draw(st.sampled_from([[24, 24], [24, 24], [24, 24]]))
    Y_shape = draw(st.sampled_from([[24, 24], [24, 24], [24, 24]]))
    trans_x = draw(st.booleans())
    trans_y = draw(st.booleans())

    matmul_v2_op = OpConfig(
        type = "matmul_v2",
        inputs = {"X" : ["input_data_x"], "Y" : ["input_data_y"]},
        outputs = {"Out": ["output_data"]},
        attrs = {"trans_x":trans_x, "trans_y":trans_y})
    program_config = ProgramConfig(
        ops=[matmul_v2_op],
        weights={},
        inputs={
            "input_data_x":
            TensorConfig(shape=X_shape),
            "input_data_y" : TensorConfig(shape=Y_shape)
        },
        outputs={"output_data"})
    return program_config
