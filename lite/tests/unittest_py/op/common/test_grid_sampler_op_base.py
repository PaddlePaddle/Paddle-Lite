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
    in_shape1 = draw(
        st.lists(
            st.integers(
                min_value=10, max_value=100), min_size=4, max_size=4))

    in_shape2 = []
    in_shape2.append(in_shape1[0])
    in_shape2.append(in_shape1[2])
    in_shape2.append(in_shape1[3])
    in_shape2.append(2)

    align_corners = draw(st.booleans())
    mode = draw(st.sampled_from(["bilinear", "nearest"]))
    padding_mode = draw(st.sampled_from(["zeros", "reflection", "border"]))

    grid_sampler_op = OpConfig(
        type="grid_sampler",
        inputs={"X": ["input_data"],
                "Grid": ["grid_data"]},
        outputs={"Output": ["output_data"]},
        attrs={
            "align_corners": align_corners,
            "mode": mode,
            "padding_mode": padding_mode
        })

    program_config = ProgramConfig(
        ops=[grid_sampler_op],
        weights={"grid_data": TensorConfig(shape=in_shape2)},
        inputs={"input_data": TensorConfig(shape=in_shape1)},
        outputs=["output_data"])

    return program_config
