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
    transpose1_input_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=10), min_size=4, max_size=4))
    axis_dim4 = draw(
        st.lists(
            st.integers(
                min_value=0, max_value=3), min_size=4, max_size=4))
    assume(sorted(axis_dim4) == [0, 1, 2, 3])
    axis_dim4_1 = [
        axis_dim4.index(0), axis_dim4.index(1), axis_dim4.index(2),
        axis_dim4.index(3)
    ]
    transpose_type = draw(st.sampled_from(["transpose", "transpose2"]))

    if transpose_type == "transpose":
        transpose1_op = OpConfig(
            type="transpose",
            inputs={"X": ["transpose1_input_x"]},
            outputs={"Out": ["transpose1_output"]},
            attrs={"axis": axis_dim4})

        softmax_op = OpConfig(
            type="softmax",
            inputs={"X": ["transpose1_output"]},
            outputs={"Out": ["softmax_output"]},
            attrs={"axis": -1})

        transpose2_op = OpConfig(
            type="transpose",
            inputs={"X": ["softmax_output"]},
            outputs={"Out": ["output_data"]},
            attrs={"axis": axis_dim4_1})
    elif transpose_type == "transpose2":
        transpose1_op = OpConfig(
            type="transpose2",
            inputs={"X": ["transpose1_input_x"]},
            outputs={
                "Out": ["transpose1_output"],
                "XShape": ["transpose1_XShape"]
            },
            attrs={"axis": axis_dim4})

        softmax_op = OpConfig(
            type="softmax",
            inputs={"X": ["transpose1_output"]},
            outputs={"Out": ["softmax_output"]},
            attrs={"axis": -1})

        transpose2_op = OpConfig(
            type="transpose2",
            inputs={"X": ["softmax_output"]},
            outputs={"Out": ["output_data"],
                     "XShape": ["transpose2_XShape"]},
            attrs={"axis": axis_dim4_1})

    ops = [transpose1_op, softmax_op, transpose2_op]
    program_config = ProgramConfig(
        ops=ops,
        weights={},
        inputs={
            "transpose1_input_x": TensorConfig(shape=transpose1_input_shape)
        },
        outputs=["output_data"])
    return program_config
