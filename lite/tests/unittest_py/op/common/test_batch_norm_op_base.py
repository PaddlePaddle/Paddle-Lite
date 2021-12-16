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
    is_test_val = draw(st.sampled_from([True, False]))
    epsilon = draw(st.floats(min_value=0.00001, max_value=0.001))
    momentum = draw(st.floats(min_value=0.1, max_value=0.9))

    def generate_input(*args, **kwargs):
        return np.random.random(in_shape).astype(np.float32)

    def generate_scale(*args, **kwargs):
        return np.random.random([in_shape[1]]).astype(np.float32) + 0.5

    def generate_bias(*args, **kwargs):
        return np.random.random([in_shape[1]]).astype(np.float32)

    def generate_mean(*args, **kwargs):
        return np.random.random([in_shape[1]]).astype(np.float32)

    def generate_variance(*args, **kwargs):
        return np.random.random([in_shape[1]]).astype(np.float32)

    batch_norm_ops = OpConfig(
        type="batch_norm",
        inputs={
            "X": ["input_data"],
            "Scale": ["scale_data"],
            "Bias": ["bias_data"],
            "Mean": ["mean_data"],
            "Variance": ["variance_data"]
        },
        outputs={
            "Y": ["output_data"],
            "MeanOut": ["mean_data"],
            "VarianceOut": ["variance_data"],
            "SavedMean": ["saved_mean"],
            "SavedVariance": ["saved_variance"]
        },
        attrs={
            "is_test": False,
            "trainable_statistics": False,
            "data_layout": "NCHW",
            "use_global_stats": False,
            "epsilon": epsilon,
            "momentum": momentum
        })
    program_config = ProgramConfig(
        ops=[batch_norm_ops],
        weights={},
        inputs={
            "input_data": TensorConfig(data_gen=partial(generate_input)),
            "scale_data": TensorConfig(data_gen=partial(generate_scale)),
            "bias_data": TensorConfig(data_gen=partial(generate_bias)),
            "mean_data": TensorConfig(data_gen=partial(generate_mean)),
            "variance_data": TensorConfig(data_gen=partial(generate_variance)),
        },
        outputs=[
            "output_data", "mean_data", "variance_data", "saved_mean",
            "saved_variance"
        ])
    return program_config
