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
import copy


def sample_program_configs(draw):
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=2, max_value=100), min_size=2, max_size=2))
    id_shape = draw(
        st.lists(
            st.integers(
                min_value=2, max_value=100), min_size=2, max_size=3))
    pidx = draw(st.sampled_from([-1, 0, 1, 2]))
    op_type_str = draw(st.sampled_from(["lookup_table", "lookup_table_v2"]))

    def generate_input(*args, **kwargs):
        return np.random.random(in_shape).astype(np.float32)

    def generate_ids(*args, **kwargs):
        extend_id = copy.deepcopy(id_shape)
        extend_id.append(1)
        return np.random.randint(
            low=0, high=in_shape[0], size=extend_id).astype(np.int64)

    build_ops = OpConfig(
        type=op_type_str,
        inputs={
            "W": ["w_data"],
            "Ids": ["ids_data"],
        },
        outputs={"Out": ["output_data"], },
        attrs={"padding_idx": int(pidx), })
    program_config = ProgramConfig(
        ops=[build_ops],
        weights={},
        inputs={
            "w_data": TensorConfig(data_gen=partial(generate_input)),
            "ids_data": TensorConfig(data_gen=partial(generate_ids)),
        },
        outputs=["output_data"])
    return program_config
