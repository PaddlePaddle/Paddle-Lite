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
    input_type = "float32" # draw(st.sampled_from(["int32", "int64", "float32"]))
    index_type = "int64" # draw(st.sampled_from(["int32", "int64"]))

    in_shape = [2, 3, 4] # [3, 5, 9, 10] # draw(st.lists(st.integers(min_value=2, max_value=8), min_size=4, max_size=4))
    index_shape = [2, 2] # [3, 2] # draw(st.lists(st.integers(min_value=1, max_value=3), min_size=2, max_size=4))
    assume(index_shape[-1] <= len(in_shape))
    update_shape = [2, 4] # [3, 9, 10] # index_shape[:-1] + in_shape[index_shape[-1]:]

    def generate_data(*args, **kwargs):
        if kwargs["type"] == "int32":
            return np.random.randint(kwargs["low"], kwargs["high"], kwargs["shape"]).astype(np.int32)
        elif kwargs["type"] == "int64":
            return np.random.randint(kwargs["low"], kwargs["high"], kwargs["shape"]).astype(np.int64)
        elif kwargs["type"] == "float32":
            return (kwargs["high"] - kwargs["low"]) * np.random.random(kwargs["shape"]).astype(np.float32) + kwargs["low"]

    def generate_index_data(*args, **kwargs):
        if kwargs["type"] == "int32":
            index_data = np.array([[0, 1],
                        [1, 2]]).astype(np.int32)
            return index_data
        elif kwargs["type"] == "int64":
            index_data = np.array([[0, 1],
                        [1, 2]]).astype(np.int64)
            return index_data

    scatter_nd_add_op = OpConfig(
        type = "scatter_nd_add",
        inputs = {"X" : ["input_data"], "Index" : ["index"], "Updates" : ["updates"]},
        outputs = {"Out" : ["output_data"]},
        attrs = {})

    program_config = ProgramConfig(
        ops=[scatter_nd_add_op],
        weights={},
        inputs={
            "input_data" : TensorConfig(data_gen=partial(generate_data, type=input_type, low=-10, high=10, shape=in_shape)),
            "index" : TensorConfig(data_gen=partial(generate_index_data, type=index_type)),
            "updates" : TensorConfig(data_gen=partial(generate_data, type=input_type, low=-10, high=10, shape=update_shape)),
        },
        outputs=["output_data"])

    return program_config
