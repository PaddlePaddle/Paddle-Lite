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
    reshape1_input_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=128), min_size=4, max_size=4))
    reshape1_output_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=128), min_size=5, max_size=5))
    reshape2_output_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=128), min_size=4, max_size=4))

    reshape1_output_shape[0] = reshape1_input_shape[0]
    reshape1_output_shape[3] = reshape1_input_shape[2]
    reshape1_output_shape[4] = reshape1_input_shape[3]

    assume(reshape1_output_shape[1] * reshape1_output_shape[2] ==
           reshape1_input_shape[1])

    reshape2_output_shape[0] = reshape1_input_shape[0]
    reshape2_output_shape[1] = -1
    reshape2_output_shape[2] = reshape1_input_shape[2]
    reshape2_output_shape[3] = reshape1_input_shape[3]

    reshape1_op = OpConfig(
        type="reshape",
        inputs={"X": ["reshape1_input_x"]},
        outputs={"Out": ["reshape1_output"]},
        attrs={"shape": reshape1_output_shape})

    transpose_op = OpConfig(
        type="transpose",
        inputs={"X": ["reshape1_output"]},
        outputs={"Out": ["transpose_output"]},
        attrs={"use_mkldnn": False,
               "axis": [0, 2, 1, 3, 4]})

    reshape2_op = OpConfig(
        type="reshape",
        inputs={"X": ["transpose_output"]},
        outputs={"Out": ["output_data"]},
        attrs={"shape": reshape2_output_shape})

    ops = [reshape1_op, transpose_op, reshape2_op]
    program_config = ProgramConfig(
        ops=ops,
        weights={},
        inputs={"reshape1_input_x": TensorConfig(shape=reshape1_input_shape)},
        outputs=["output_data"])
    return program_config
