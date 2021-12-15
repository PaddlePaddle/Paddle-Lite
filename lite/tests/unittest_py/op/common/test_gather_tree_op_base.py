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
    ids_shape = draw(st.sampled_from([[3, 2, 2], [10, 9, 12], [4, 4, 3]]))
    ids_size = ids_shape[0] * ids_shape[1] * ids_shape[2]

    index_type = draw(st.sampled_from(["int32", "int64"]))

    def generate_ids_data(*args, **kwargs):
        if index_type == "int32":
            return np.random.randint(0, ids_size, ids_shape).astype(np.int32)
        else:
            return np.random.randint(0, ids_size, ids_shape).astype(np.int64)

    def generate_parents_data(*args, **kwargs):
        if index_type == "int32":
            return np.random.randint(0, ids_shape[2],
                                     ids_shape).astype(np.int32)
        else:
            return np.random.randint(0, ids_shape[2],
                                     ids_shape).astype(np.int64)

    gather_tree_op = OpConfig(
        type="gather_tree",
        inputs={"Ids": ["ids_data"],
                "Parents": ["parents_data"]},
        outputs={"Out": ["output_data"]},
        attrs={})
    program_config = ProgramConfig(
        ops=[gather_tree_op],
        weights={},
        inputs={
            "ids_data": TensorConfig(data_gen=partial(generate_ids_data)),
            "parents_data":
            TensorConfig(data_gen=partial(generate_parents_data))
        },
        outputs=["output_data"])
    return program_config
