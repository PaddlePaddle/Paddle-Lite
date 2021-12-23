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
                min_value=1, max_value=10), min_size=4, max_size=4))
    epsilon = draw(st.floats(min_value=0.0001, max_value=0.0005))
    begin_norm_axis = draw(st.sampled_from([1, 2]))

    def generate_input(*args, **kwargs):
        return np.random.random(in_shape).astype(np.float32)

    channel_dim = 1
    for dim in range(begin_norm_axis, 4):
        channel_dim = channel_dim * in_shape[dim]

    def generate_scale(*args, **kwargs):
        return np.random.random([channel_dim]).astype(np.float32)

    def generate_bias(*args, **kwargs):
        return np.random.random([channel_dim]).astype(np.float32)

    run_op = OpConfig(
        type="layer_norm",
        inputs={
            "X": ["input_data"],
            "Scale": ["scale_data"],
            "Bias": ["bias_data"]
        },
        outputs={
            "Y": ["output_data"],
            "Mean": ["mean_data"],
            "Variance": ["var_data"],
        },
        attrs={"epsilon": epsilon,
               "begin_norm_axis": begin_norm_axis})
    program_config = ProgramConfig(
        ops=[run_op],
        weights={},
        inputs={
            "input_data": TensorConfig(data_gen=partial(generate_input)),
            "scale_data": TensorConfig(data_gen=partial(generate_scale)),
            "bias_data": TensorConfig(data_gen=partial(generate_bias)),
        },
        outputs=["output_data", "mean_data", "var_data"])
    return program_config
