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


def sample_program_configs(draw):
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=4, max_value=8), min_size=3, max_size=4))
    value0 = draw(st.integers(min_value=0, max_value=in_shape[0] - 1))
    value1 = draw(st.integers(min_value=0, max_value=in_shape[1] - 1))
    value2 = draw(st.integers(min_value=0, max_value=in_shape[2] - 1))
    index = draw(
        st.sampled_from([[value0], [value0, value1], [value0, value1, value2]
                         ]))
    index_type = draw(st.sampled_from(["int32", "int64"]))

    def generate_index(*args, **kwargs):
        if index_type == "int32":
            return np.array(index).astype(np.int32)
        else:
            return np.array(index).astype(np.int64)

    def generate_input_int32(*args, **kwargs):
        return np.random.random(in_shape).astype(np.int32)

    def generate_input_int64(*args, **kwargs):
        return np.random.random(in_shape).astype(np.int64)

    def generate_input_float32(*args, **kwargs):
        return np.random.random(in_shape).astype(np.float32)

    generate_input = draw(
        st.sampled_from([
            generate_input_int32, generate_input_int64, generate_input_float32
        ]))

    op_inputs = {"X": ["input_data"], "Index": ["index_data"]}
    program_inputs = {
        "input_data": TensorConfig(data_gen=partial(generate_input)),
        "index_data": TensorConfig(data_gen=partial(generate_index))
    }

    gather_nd_op = OpConfig(
        type="gather_nd",
        inputs=op_inputs,
        outputs={"Out": ["output_data"]},
        attrs={"axis": 1})
    program_config = ProgramConfig(
        ops=[gather_nd_op],
        weights={},
        inputs=program_inputs,
        outputs=["output_data"])
    return program_config
