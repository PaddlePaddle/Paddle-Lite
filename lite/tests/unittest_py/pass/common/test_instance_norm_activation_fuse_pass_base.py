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
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=2, max_value=64), min_size=2, max_size=4))
    epsilon_data = draw(st.floats(min_value=0.0, max_value=0.001))

    act_type = draw(st.sampled_from(['relu', 'relu6', 'leaky_relu']))

    def generate_act_attrs(act_type_str):
        attrs = {}
        if act_type_str == 'relu6':
            attrs = {"threshold": threshold}
        if act_type_str == 'leaky_relu':
            attrs = {"alpha": alpha}
        return attrs

    instance_norm_op = OpConfig(
        type="instance_norm",
        inputs={
            "X": ["input_data"],
            "Scale": ["scale_data"],
            "Bias": ["bisa_data"]
        },
        outputs={
            "Y": ["y_output_data"],
            "SavedMean": ["SavedMean_data"],
            "SavedVariance": ["SavedVariance_data"]
        },
        attrs={"epsilon": epsilon_data})

    active_op = OpConfig(
        type=act_type,
        inputs={"X": ["y_output_data"]},
        outputs={"Out": ["output_data"]},
        attrs=generate_act_attrs(act_type))

    ops = [instance_norm_op, active_op]
    program_config = ProgramConfig(
        ops=ops,
        weights={},
        inputs={
            "input_data": TensorConfig(shape=in_shape),
            "scale_data": TensorConfig(shape=[in_shape[1]]),
            "bisa_data": TensorConfig(shape=[in_shape[1]])
        },
        outputs=["output_data"])
    return program_config
