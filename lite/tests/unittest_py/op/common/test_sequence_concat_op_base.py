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
    def generate_input(*args, **kwargs):
        if kwargs["type"] == "int64":
            return np.random.randint(kwargs["low"], kwargs["high"],
                                     kwargs["shape"]).astype(np.int64)
        elif kwargs["type"] == "float32":
            return np.random.random(kwargs["shape"]).astype(np.float32)

    input_type = draw(st.sampled_from(["int64", "float32"]))
    feature_len = draw(st.integers(min_value=3, max_value=90))
    lod_info_x1 = generate_input(
        type="int64", low=0, high=10, shape=[1, feature_len + 1])
    lod_info_x2 = generate_input(
        type="int64", low=0, high=11, shape=[1, feature_len + 1])
    lod_info_x3 = generate_input(
        type="int64", low=0, high=12, shape=[1, feature_len + 1])
    lod_info_x1 = np.sort(lod_info_x1)
    lod_info_x2 = np.sort(lod_info_x2)
    lod_info_x3 = np.sort(lod_info_x3)
    x1_lod_len = lod_info_x1[0][-1]
    x2_lod_len = lod_info_x2[0][-1]
    x3_lod_len = lod_info_x3[0][-1]
    y_lod_len = x1_lod_len + x2_lod_len + x3_lod_len
    assume(lod_info_x1[0][0] == lod_info_x2[0][0])
    assume(lod_info_x2[0][0] == lod_info_x3[0][0])

    sequence_concat_op = OpConfig(
        type="sequence_concat",
        inputs={"X": ["input1_data", "input2_data", "input3_data"]},
        outputs={"Out": ["output_data"]},
        attrs={})

    program_config = ProgramConfig(
        ops=[sequence_concat_op],
        weights={},
        inputs={
            "input1_data": TensorConfig(
                data_gen=partial(
                    generate_input,
                    type=input_type,
                    low=-10,
                    high=10,
                    shape=[x1_lod_len, feature_len]),
                lod=lod_info_x1),
            "input2_data": TensorConfig(
                data_gen=partial(
                    generate_input,
                    type=input_type,
                    low=-10,
                    high=10,
                    shape=[x2_lod_len, feature_len]),
                lod=lod_info_x2),
            "input3_data": TensorConfig(
                data_gen=partial(
                    generate_input,
                    type=input_type,
                    low=-10,
                    high=10,
                    shape=[x3_lod_len, feature_len]),
                lod=lod_info_x3),
        },
        outputs=["output_data"])

    return program_config
