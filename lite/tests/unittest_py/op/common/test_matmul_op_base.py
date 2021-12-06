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
    alpha = draw(st.sampled_from([0.1, 1.1, -1.5]))
    fused_reshape_X = draw(st.sampled_from([[]]))
    fused_reshape_Y = draw(st.sampled_from([[]]))
    fused_transpose_X = draw(st.sampled_from([[]]))
    fused_transpose_Y = draw(st.sampled_from([[]]))
    fused_reshape_Out = draw(st.sampled_from([[]]))
    fused_transpose_Out = draw(st.sampled_from([[]]))
    Scale_x = draw(st.sampled_from([0.1, 1.1]))
    Scale_y = draw(st.sampled_from([0.1, 1.1]))
    Scale_out = draw(st.sampled_from([0.1, 1.1]))
    force_fp32_output = draw(st.booleans())

    matmul_op = OpConfig(
        type = "matmul",
        inputs = {"X" : ["input_data_x"], "Y" : ["input_data_y"]},
        outputs = {"Out": ["output_data"]},
        attrs = {"transpose_X" : transpose_X, "transpose_Y" : transpose_Y, "alpha" : alpha, "fused_reshape_X" : fused_reshape_X,
            "fused_reshape_Y" : fused_reshape_Y, "fused_transpose_X" : fused_transpose_X, "fused_transpose_Y" : fused_transpose_Y,
            "fused_reshape_Out" : fused_reshape_Out, "fused_transpose_Out" : fused_transpose_Out, "Scale_x" : Scale_x, "Scale_y" : Scale_y,
            "Scale_out" : Scale_out, "force_fp32_output" : force_fp32_output})
    program_config = ProgramConfig(
        ops=[matmul_op],
        weights={},
        inputs={
            "input_data_x":
            TensorConfig(shape=X_shape),
            "input_data_y" : TensorConfig(shape=Y_shape)
        },
        outputs={"output_data"})
    return program_config
