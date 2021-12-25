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
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st


def sample_program_configs(draw):
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=6, max_value=64), min_size=4, max_size=4))

    axes = draw(st.sampled_from([[3], [0, 1], [0, 1, 2], [0, 1, 2, 3]]))
    starts = draw(st.sampled_from([[-1], [0, 1], [0, 1, 2], [0, 1, 2, 3]]))
    ends = draw(st.sampled_from([[10000], [1, 2], [1, 2, 3], [1, 2, 3, 4]]))
    decrease_axis = draw(
        st.sampled_from([[3], [0, 1], [0, 1, 2], [0, 1, 2, 3]]))
    infer_flags = draw(st.sampled_from([[1, 1, 1]]))
    input_num = draw(st.sampled_from([0, 1]))
    input_type = draw(st.sampled_from(["type_float"]))

    assume((len(starts) == len(ends)) & (len(starts) == len(axes)))
    assume(len(decrease_axis) == len(starts))
    assume(len(axes) <= len(in_shape))

    if input_num == 1:
        assume(len(in_shape) == 4)

    def generate_input(attrs: List[Dict[str, Any]], input_type):
        if input_type == "type_float":
            return np.random.normal(0.0, 1.0, in_shape).astype(np.float32)
        elif input_type == "type_int":
            return np.random.randint(in_shape).astype(np.int32)
        elif input_type == "type_int64":
            return np.random.randint(in_shape).astype(np.int64)

    def generate_starts(*args, **kwargs):
        return np.array(starts, dtype="int32")

    def generate_ends(*args, **kwargs):
        return np.array(ends, dtype="int32")

    dics = [{
        "axes": axes,
        "starts": starts,
        "ends": ends,
        "decrease_axis": decrease_axis,
        "infer_flags": [-1, -1, -1]
    }, {
        "axes": axes,
        "starts": starts,
        "ends": ends,
        "decrease_axis": decrease_axis,
        "infer_flags": infer_flags
    }, {}]

    dics_intput = [{
        "Input": ["input_data"],
        "StartsTensor": ["starts_data"],
        "EndsTensor": ["ends_data"],
    }, {
        "Input": ["input_data"]
    }, {}]

    dics_weight = [{
        "starts_data": TensorConfig(data_gen=partial(generate_starts)),
        "ends_data": TensorConfig(data_gen=partial(generate_ends))
    }, {}]

    ops_config = OpConfig(
        type="slice",
        inputs=dics_intput[input_num],
        outputs={"Out": ["output_data"]},
        attrs=dics[input_num])

    program_config = ProgramConfig(
        ops=[ops_config],
        weights=dics_weight[input_num],
        inputs={
            "input_data":
            TensorConfig(data_gen=partial(generate_input, dics, input_type))
        },
        outputs=["output_data"])

    return program_config
