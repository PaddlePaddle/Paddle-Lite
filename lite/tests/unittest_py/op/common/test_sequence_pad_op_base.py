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
        if kwargs["type"] == "int32":
            return np.random.randint(kwargs["low"], kwargs["high"],
                                     kwargs["shape"]).astype(np.int32)
        elif kwargs["type"] == "int64":
            return np.random.randint(kwargs["low"], kwargs["high"],
                                     kwargs["shape"]).astype(np.int64)
        elif kwargs["type"] == "float32":
            return (kwargs["high"] - kwargs["low"]) * np.random.random(kwargs[
                "shape"]).astype(np.float32) + kwargs["low"]

    out_dtype_dict = {
        "int32": np.int32,
        "int64": np.int64,
        "float32": np.float32
    }

    input_type = draw(st.sampled_from(["int32", "int64", "float32"]))
    x_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=10), min_size=2, max_size=7))
    x_len_lod = generate_input(
        type="int64", low=0, high=10, shape=[1, len(x_shape)])
    x_len_lod = np.sort(x_len_lod)
    x_len_lod[-1] = x_shape[0]

    padded_length = len(x_shape)
    pad_value_shape = [1]

    # assume
    time_step_shape = x_shape[1:]
    assume(len(x_shape) >= 2)
    assume(len(pad_value_shape) == 1 or pad_value_shape == time_step_shape)
    assume(len(np.array(x_len_lod).shape) >= 2)

    # assume
    seq_num = len(x_len_lod[0]) - 1
    max_seq_len = 0
    for i in range(0, seq_num):
        max_seq_len = max(max_seq_len, x_len_lod[0][i + 1] - x_len_lod[0][i])
    real_padded_length = padded_length
    if real_padded_length == -1:
        real_padded_length = max_seq_len
    assume(real_padded_length >= max_seq_len)

    sequence_pad_op = OpConfig(
        type="sequence_pad",
        inputs={"X": ["x_data"],
                "PadValue": ["pad_value"]},
        outputs={"Out": ["output_data"],
                 "Length": ["length_data"]},
        attrs={"padded_length": padded_length},
        outputs_dtype={
            "output_data": out_dtype_dict[input_type],
            "length_data": np.int64
        })

    program_config = ProgramConfig(
        ops=[sequence_pad_op],
        weights={},
        inputs={
            "x_data": TensorConfig(
                data_gen=partial(
                    generate_input,
                    type=input_type,
                    low=-10,
                    high=10,
                    shape=x_shape),
                lod=x_len_lod),
            "pad_value": TensorConfig(data_gen=partial(
                generate_input,
                type=input_type,
                low=0,
                high=10,
                shape=pad_value_shape))
        },
        outputs=["output_data", "length_data"])

    return program_config
