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
from hypothesis import assume


def sample_program_configs(draw):
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=10, max_value=20), min_size=4, max_size=4))
    lod_data = draw(
        st.sampled_from([[0, 3, 5, in_shape[0]], [0, 4, 7, in_shape[0]],
                         [0, 4, in_shape[0]], [0, 7, in_shape[0]]]))
    lod_data1 = draw(
        st.sampled_from([[0, 3, 5, in_shape[0]], [0, 4, 7, in_shape[0]],
                         [0, 4, in_shape[0]], [0, 7, in_shape[0]]]))
    lod_data2 = draw(
        st.sampled_from([[0, 3, 5, in_shape[0]], [0, 4, 7, in_shape[0]],
                         [0, 4, in_shape[0]], [0, 7, in_shape[0]]]))
    case_num = draw(st.sampled_from([0, 1]))

    def generate_input_x(*args, **kwargs):
        return np.random.random(in_shape).astype(np.float32)

    def generate_input_y(*args, **kwargs):
        return np.array(lod_data1).astype(np.int32)

    if case_num == 0:
        build_ops = OpConfig(
            type="lod_reset",
            inputs={"X": ["input_data_x"],
                    "Y": []},
            outputs={"Out": ["output_data"], },
            attrs={"target_lod": lod_data,
                   'append': True})
        program_config = ProgramConfig(
            ops=[build_ops],
            weights={},
            inputs={
                "input_data_x": TensorConfig(data_gen=partial(
                    generate_input_x, lod=list(lod_data2))),
            },
            outputs=["output_data"])
    elif case_num == 1:
        build_ops = OpConfig(
            type="lod_reset",
            inputs={"X": ["input_data_x"],
                    "Y": ["input_data_y"]},
            outputs={"Out": ["output_data"], },
            attrs={"target_lod": [],
                   'append': True})
        program_config = ProgramConfig(
            ops=[build_ops],
            weights={},
            inputs={
                "input_data_x": TensorConfig(data_gen=partial(
                    generate_input_x, lod=list(lod_data2))),
                "input_data_y":
                TensorConfig(data_gen=partial(generate_input_y)),
            },
            outputs=["output_data"])
    return program_config
