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
    input1_shape = draw(
        st.lists(
            st.integers(min_value=1), min_size=4, max_size=4))
    input2_shape = draw(
        st.lists(
            st.integers(min_value=1), min_size=4, max_size=4))
    pad_size = draw(st.integers(min_value=0))
    kernel_size = draw(st.integers(min_value=1))
    max_displacement = draw(st.integers(min_value=1))
    stride1 = draw(st.integers(min_value=1))
    stride2 = draw(st.integers(min_value=1))
    corr_type_multiply = draw(st.integers(min_value=1))

    correlation_op = OpConfig(
        type="correlation",
        inputs={"Input1": ["input1"],
                "Input2": ["input2"]},
        outputs={"Output": ["output"]},
        attrs={
            "pad_size": pad_size,
            "kernel_size": kernel_size,
            "max_displacement": max_displacement,
            "stride1": stride1,
            "stride2": stride2,
            "corr_type_multiply": corr_type_multiply
        })

    program_config = ProgramConfig(
        ops=[correlation_op],
        weights={},
        inputs={
            "input1": TensorConfig(shape=input1_shape),
            "input2": TensorConfig(shape=input2_shape),
        },
        outputs=["output"])
    return program_config
