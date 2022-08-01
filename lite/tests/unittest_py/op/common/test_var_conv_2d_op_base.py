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
from hypothesis import assume
import hypothesis.strategies as st


def sample_program_configs(draw):

    COLUMN_shape = draw(
        st.lists(
            st.integers(
                min_value=6, max_value=6), min_size=2, max_size=2))
    ROW_shape = draw(
        st.lists(
            st.integers(
                min_value=6, max_value=6), min_size=2, max_size=2))
    # Only when the width and length are equal，there is no diff
    #COLUMN_shape = draw(st.sampled_from([[5,4]]))
    #ROW_shape = draw(st.sampled_from([[6,7]]))

    OutputChannel_data = draw(st.integers(min_value=3, max_value=8))
    InputChannel_data = draw(st.integers(min_value=3, max_value=8))
    KernelH_data = draw(st.integers(min_value=3, max_value=3))
    KernelW_data = draw(st.integers(min_value=3, max_value=3))
    StrideH_data = draw(st.integers(min_value=1, max_value=1))
    StrideW_data = draw(st.integers(min_value=1, max_value=1))

    W_shape = draw(st.lists(st.integers(), min_size=2, max_size=2))
    W_shape[1] = InputChannel_data * KernelH_data * KernelW_data
    W_shape[0] = OutputChannel_data

    X_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=50), min_size=2, max_size=2))
    X_shape[1] = 1
    X_shape[0] = (
        COLUMN_shape[0] * ROW_shape[0] + COLUMN_shape[1] * ROW_shape[1]
    ) * InputChannel_data

    # lod_data1 must have three elements in lite
    lod_data1 = draw(
        st.sampled_from([[[
            0, COLUMN_shape[0] * ROW_shape[1] * InputChannel_data, X_shape[0]
        ], [0, COLUMN_shape[0], sum(COLUMN_shape)],
                          [0, ROW_shape[0], sum(ROW_shape)]]]))
    # lod_data2 and lod_data3 is only used in paddle
    lod_data2 = draw(
        st.sampled_from([[[0, COLUMN_shape[0], sum(COLUMN_shape)]]]))
    lod_data3 = draw(st.sampled_from([[[0, ROW_shape[0], sum(ROW_shape)]]]))

    var_conv2d_op = OpConfig(
        type="var_conv_2d",
        inputs={
            "X": ["X_data"],
            "COLUMN": ["COLUMN_data"],
            "ROW": ["ROW_data"],
            "W": ["W_data"]
        },
        outputs={"Out": ["Out_data"],
                 "Col": ["Col_data"]},
        attrs={
            "OutputChannel": OutputChannel_data,
            "InputChannel": InputChannel_data,
            "KernelH": KernelH_data,
            "KernelW": KernelW_data,
            "StrideH": StrideH_data,
            "StrideW": StrideW_data,
        })
    program_config = ProgramConfig(
        ops=[var_conv2d_op],
        weights={},
        inputs={
            "X_data": TensorConfig(
                shape=X_shape, lod=lod_data1),
            "COLUMN_data": TensorConfig(
                shape=COLUMN_shape, lod=lod_data2),
            "ROW_data": TensorConfig(
                shape=ROW_shape, lod=lod_data3),
            "W_data": TensorConfig(shape=W_shape)
        },
        outputs=["Out_data", "Col_data"])
    return program_config
