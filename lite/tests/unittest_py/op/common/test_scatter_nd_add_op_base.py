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
    in_shape = draw(st.lists(st.integers(min_value=1, max_value=8), min_size=1, max_size=4))
    index_shape = draw(st.lists(st.integers(min_value=1, max_value=8), min_size=2, max_size=4))
    print('1')
    print('len(index_shape): ', len(index_shape))
    assume(len(index_shape) <= len(in_shape))
    print('2')

    update_shape = index_shape[:-1] + in_shape[index_shape[-1]:]

    input_type = draw(st.sampled_from(["int32", "int64", "float32"]))
    index_type = draw(st.sampled_from(["int32", "int64"]))
    update_type = input_type

    print('in_shape: ', in_shape)
    print('index_shape: ', index_shape)
    print('update_shape: ', update_shape)

    def generate_input_int32(*args, **kwargs):
        print('input int32: ', np.random.randint(in_shape).astype(np.int32))
        return np.random.randint(in_shape).astype(np.int32)
    def generate_input_int64(*args, **kwargs):
        return np.random.randint(in_shape).astype(np.int64)
    def generate_input_float32(*args, **kwargs):
        return np.random.random(in_shape).astype(np.float32)

    def generate_index_int32(*args, **kwargs):
        print("index int32:", np.random.randint(index_shape).astype(np.int32))
        return np.random.randint(index_shape).astype(np.int32)
    def generate_index_int64(*args, **kwargs):
        print("index int64:", np.random.randint(index_shape).astype(np.int64))
        return np.random.randint(index_shape).astype(np.int64)

    def generate_update_int32(*args, **kwargs):
        return np.random.randint(update_shape).astype(np.int32)
    def generate_update_int64(*args, **kwargs):
        return np.random.randint(update_shape).astype(np.int64)
    def generate_update_float32(*args, **kwargs):
        return np.random.random(update_shape).astype(np.float32)

    scatter_nd_add_op = OpConfig(
        type = "scatter_nd_add",
        inputs = {"X" : ["input_data"], "Index" : ["index"], "Updates" : ["updates"]},
        outputs = {"Out" : ["output_data"]},
        attrs = {})

    if input_type == "int32":
        if index_type == "int32":
            program_config = ProgramConfig(
                ops=[scatter_nd_add_op],
                weights={},
                inputs={
                    "input_data" : TensorConfig(data_gen=partial(generate_input_int32)),
                    "index" : TensorConfig(data_gen=partial(generate_index_int32)),
                    "updates" : TensorConfig(data_gen=partial(generate_update_int32))
                },
                outputs=["output_data"])
        elif index_type == "int64":
            program_config = ProgramConfig(
                ops=[scatter_nd_add_op],
                weights={},
                inputs={
                    "input_data" : TensorConfig(data_gen=partial(generate_input_int32)),
                    "index" : TensorConfig(data_gen=partial(generate_index_int64)),
                    "updates" : TensorConfig(data_gen=partial(generate_update_int32))
                },
                outputs=["output_data"])
    elif input_type == "int64":
        if index_type == "int32":
            program_config = ProgramConfig(
                ops=[scatter_nd_add_op],
                weights={},
                inputs={
                    "input_data" : TensorConfig(data_gen=partial(generate_input_int64)),
                    "index" : TensorConfig(data_gen=partial(generate_index_int32)),
                    "updates" : TensorConfig(data_gen=partial(generate_update_int64))
                },
                outputs=["output_data"])
        elif index_type == "int64":
            program_config = ProgramConfig(
                ops=[scatter_nd_add_op],
                weights={},
                inputs={
                    "input_data" : TensorConfig(data_gen=partial(generate_input_int64)),
                    "index" : TensorConfig(data_gen=partial(generate_index_int64)),
                    "updates" : TensorConfig(data_gen=partial(generate_update_int64))
                },
                outputs=["output_data"])
    elif input_type == "float32":
        if index_type == "int32":
            program_config = ProgramConfig(
                ops=[scatter_nd_add_op],
                weights={},
                inputs={
                    "input_data" : TensorConfig(data_gen=partial(generate_input_float32)),
                    "index" : TensorConfig(data_gen=partial(generate_index_int32)),
                    "updates" : TensorConfig(data_gen=partial(generate_update_float32))
                },
                outputs=["output_data"])
        elif index_type == "int64":
            program_config = ProgramConfig(
                ops=[scatter_nd_add_op],
                weights={},
                inputs={
                    "input_data" : TensorConfig(data_gen=partial(generate_input_float32)),
                    "index" : TensorConfig(data_gen=partial(generate_index_int64)),
                    "updates" : TensorConfig(data_gen=partial(generate_update_float32))
                },
                outputs=["output_data"])

    return program_config
