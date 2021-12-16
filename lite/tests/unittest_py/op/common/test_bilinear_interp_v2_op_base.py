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
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=10), min_size=4, max_size=4))
    out_size = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=10), min_size=2, max_size=2))
    out_size_shape = draw(st.sampled_from([[1, 2]]))
    align_corners = draw(st.booleans())
    align_mode = draw(st.sampled_from([0, 1]))
    out_h = draw(st.integers(min_value=1, max_value=10))
    out_w = draw(st.integers(min_value=1, max_value=10))
    scale = draw(
        st.lists(
            st.floats(
                min_value=0.1, max_value=0.9), min_size=2, max_size=2))

    def generate_input(*args, **kwargs):
        return np.random.random(in_shape).astype(np.float32)

    def generate_out_size(*args, **kwargs):
        return np.random.random(out_size_shape).astype(np.int32)

    def generate_size_tensor(*args, **kwargs):
        return np.random.randint(1, 10, [1]).astype(np.int32)

    def generate_scale(*args, **kwargs):
        return np.random.random([1]).astype(np.int32)

    bilinear_interp_v2_op = OpConfig(
        type="bilinear_interp_v2",
        inputs={
            "X": ["input_data"],
            "OutSize": ["out_size_data"],
            "SizeTensor": ["size_tensor_data1", "size_tensor_data2"],
            "Scale": ["scale_data"]
        },
        outputs={"Out": ["output_data"]},
        attrs={
            "data_layout": "NCHW",
            "out_d": 0,
            "out_h": out_h,
            "out_w": out_w,
            "scale": scale,
            "interp_method": "bilinear",
            "align_corners": align_corners,
            "align_mode": align_mode
        })
    program_config = ProgramConfig(
        ops=[bilinear_interp_v2_op],
        weights={},
        inputs={
            "input_data": TensorConfig(data_gen=partial(generate_input)),
            "out_size_data": TensorConfig(data_gen=partial(generate_out_size)),
            "size_tensor_data1":
            TensorConfig(data_gen=partial(generate_size_tensor)),
            "size_tensor_data2":
            TensorConfig(data_gen=partial(generate_size_tensor)),
            "scale_data": TensorConfig(data_gen=partial(generate_scale))
        },
        outputs=["output_data"])
    return program_config
