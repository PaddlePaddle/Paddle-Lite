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
                min_value=1, max_value=8), min_size=4, max_size=4))
    group = draw(st.integers(min_value=1, max_value=4))
    with_channel = draw(st.sampled_from([True, False]))

    assume(in_shape[1] % int(group) == 0)

    def generate_input(*args, **kwargs):
        return np.random.random(in_shape).astype(np.float32)

    def generate_scale(*args, **kwargs):
        return np.random.random([in_shape[1]]).astype(np.float32) + 0.5

    def generate_bias(*args, **kwargs):
        return np.random.random([in_shape[1]]).astype(np.float32)

    def generate_attr(with_channel):
        attrs = {}
        if with_channel == True:
            attrs = {
                "data_layout": "NCHW",
                "epsilon": float(1e-5),
                "groups": int(group)
            }
        else:
            attrs = {
                "data_layout": "NCHW",
                "channels": in_shape[1],
                "epsilon": float(1e-5),
                "groups": int(group)
            }
        return attrs

    build_ops = OpConfig(
        type="group_norm",
        inputs={
            "X": ["input_data"],
            "Scale": ["scale_data"],
            "Bias": ["bias_data"]
        },
        outputs={
            "Y": ["output_data"],
            "Mean": ["mean1"],
            "Variance": ["var1"]
        },
        attrs=generate_attr(with_channel))
    program_config = ProgramConfig(
        ops=[build_ops],
        weights={},
        inputs={
            "input_data": TensorConfig(data_gen=partial(generate_input)),
            "scale_data": TensorConfig(data_gen=partial(generate_scale)),
            "bias_data": TensorConfig(data_gen=partial(generate_bias)),
        },
        outputs=["output_data", "mean1", "var1"])
    return program_config
