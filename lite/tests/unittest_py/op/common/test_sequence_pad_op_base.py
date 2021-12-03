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
            return np.random.randint(kwargs["low"], kwargs["high"], kwargs["shape"]).astype(np.int32)
        elif kwargs["type"] == "int64":
            return np.random.randint(kwargs["low"], kwargs["high"], kwargs["shape"]).astype(np.int64)
        elif kwargs["type"] == "float32":
            return np.random.random(kwargs["shape"]).astype(np.float32)


    input_type = draw(st.sampled_from(["int32", "int64", "float32"]))

    x_shape = [9, 2, 3, 4] # draw(st.lists(st.integers(min_value=1, max_value=10), min_size=2, max_size=2))
    x_len_lod = [[0, 2, 5, x_shape[0]]]
    padded_length = 4

    sequence_pad_op = OpConfig(
        type = "sequence_pad",
        inputs = {"X" : ["x_data"], "PadValue" : ["pad_value"]},
        outputs = {"Out" : ["output_data"], "Length": ["length_data"]},
        attrs = {"padded_length" : padded_length})

    program_config = ProgramConfig(
        ops=[sequence_pad_op],
        weights={},
        inputs={
            "x_data":
            TensorConfig(data_gen=partial(generate_input, type=input_type, low=-10, high=10, shape=x_shape), lod=x_len_lod),
            "pad_value":
            TensorConfig(data_gen=partial(generate_input, type=input_type, low=0, high=10, shape=[1]))
        },
        outputs=["output_data", "length_data"])

    return program_config
